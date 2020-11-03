import torch
from .fp16.fused_optimizer import FP16_Optimizer


class OptimizerSwapper():
    def __init__(self, init_optimizer, num_partitions=1):
        self.optimizer = init_optimizer
        self.num_partitions = num_partitions
        print(f'optimizer swapper init, optimizer type: {type(init_optimizer)}')
        assert isinstance(FP16_Optimizer, self.optimizer)

        #for i, param_group in enumerate(self.optimizer.param_groups):
        #    params = param_group['params'] #.numel()
        #    assert len(params) == 1, 'expected only 1 flat param'
        #    param_count = params[0].numel()
        #    print(f'group={i}, param_count={param_count}')

        self.fp16_groups = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])

            all_partitions = self.get_data_parallel_partitions(self.fp16_groups_flat[i])

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

    def swap_in_partition(self, idx):
        pass

    def swap_out_partition(self, idx):
        pass

    def step(self):
        for idx in self.num_partitions:
            self.swap_in_partition(idx)
            self.optimizer.step()
            self.swap_out_partition(idx)

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
