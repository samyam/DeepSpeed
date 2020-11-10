import torch
from ..ops.adam.fused_adam import FusedAdam
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class OptimizerSwapper():
    def __init__(self, init_optimizer, num_partitions=2):
        self.optimizer = init_optimizer
        self.num_partitions = num_partitions
        print(f'optimizer swapper init, optimizer type: {type(self.optimizer)}')
        assert isinstance(self.optimizer, FusedAdam)

        # print(f'self.optimizer.state_dict()={self.optimizer.state_dict()}')

        self.orig_param_groups = []
        self.param_partitions = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                print(f'pre-partition p={p.shape}')

            # Save shallow copy of original params
            self.orig_param_groups.append(group["params"].copy())

            # Flatten/partition params, and move to cpu
            self.param_partitions.append(
                self.get_partitions(
                    _flatten_dense_tensors(group['params']).to('cpu'),
                    self.num_partitions))

        # Init optimizer state for each partition
        self.optimizer_states = []  # group_idx -> partition-idx -> {state-dict}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            self.optimizer_states.append([])
            for partition_idx in range(self.num_partitions):
                self.optimizer_states[group_idx].append({})
                # Swap in param partition and create zero grad
                p = self.param_partitions[group_idx][partition_idx].to('cuda')
                p.grad = torch.zeros_like(p)
                group['params'] = [p]

                #print(f'self.optimizer.state={self.optimizer.state}')
                self.optimizer.state[p] = {}

                self.optimizer.step()

                # Clear grad
                group['params'][0].grad = None

                # swap out param partition
                self.param_partitions[group_idx][partition_idx].to('cpu')

                #clear out params
                group['params'] = []

                # Swap out optimizer states
                for key, state in self.optimizer.state[p].items():
                    #print(f'key={key}, state={state}')
                    self.optimizer_states[group_idx][partition_idx][key] = state.to(
                        'cpu')
                    # Clear optimizer state
                    self.optimizer.state = {}  #[p][key] = None

    def swap_in_partition(self, partition_idx):
        for group_idx, group in enumerate(self.optimizer.param_groups):
            # swap in p.data
            self.param_partitions[group_idx][partition_idx] = self.param_partitions[
                group_idx][partition_idx].to('cuda')
            p = self.param_partitions[group_idx][partition_idx]
            # swap in p.grad
            p.grad = self.flat_grad_partitions[group_idx][partition_idx].to('cuda')
            group['params'] = [p]
            # swap in optim states
            for key, state in self.optimizer_states[group_idx][partition_idx].items():
                if p not in self.optimizer.state:
                    # create param key for optim state
                    self.optimizer.state[p] = {}
                self.optimizer.state[p][key] = state.to('cuda')

    def swap_out_partition(self, partition_idx):
        # swap out optim states
        for group_idx, group in enumerate(self.optimizer.param_groups):
            # leave param partition on cuda for now
            p = self.param_partitions[group_idx][partition_idx]
            # Clear p.grad state
            p.grad = None

            # Clear out params
            group['params'] = []

            # Clear optim states
            for key, state in self.optimizer.state[p].items():
                # print(f'key={key}, state={state}')
                self.optimizer_states[group_idx][partition_idx][key] = state.to('cpu')
                self.optimizer.state = {}

    def step(self):
        # g_32.append(p.grad.data)
        # p_32.append(p.data)
        # m_32.append(state['exp_avg'])
        # v_32.append(state['exp_avg_sq'])

        # flatten/partition p.grad
        self.flat_grad_partitions = []
        for group_idx, group_params in enumerate(self.orig_param_groups):
            pgrads = [p.grad for p in group_params]
            # print(f'group_idx={group_idx}, pgrads={pgrads}')
            self.flat_grad_partitions.append(
                self.get_partitions(_flatten_dense_tensors(pgrads),
                                    self.num_partitions))

        for idx in range(self.num_partitions):
            self.swap_in_partition(idx)
            self.optimizer.step()
            self.swap_out_partition(idx)

        # Copy updated params back
        for group_idx, _ in enumerate(self.optimizer.param_groups):
            updated_params = _unflatten_dense_tensors(
                torch.cat(self.param_partitions[group_idx]),
                self.orig_param_groups[group_idx])
            for p, q in zip(self.orig_param_groups[group_idx], updated_params):
                p.data.copy_(q.data)

            # swap param partitions back out
            for partition_idx in range(self.num_partitions):
                self.param_partitions[group_idx][partition_idx] = self.param_partitions[
                    group_idx][partition_idx].to('cpu')

    def get_partitions(self, tensor, num_partitions):
        partitions = []

        total_num_elements = tensor.numel()

        base_size = total_num_elements // num_partitions
        remaining = total_num_elements % num_partitions

        start = 0
        for idx in range(num_partitions):
            partition_size = base_size
            if idx < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def backward(self, loss):
        self.optimizer.backward(loss)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_grads_to_None=True):
        self.optimizer.zero_grad()

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)
