import torch
from .fp16.fused_optimizer import FP16_Optimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class OptimizerSwapper():
    def __init__(self, init_optimizer, num_partitions=2):
        self.optimizer = init_optimizer
        self.num_partitions = num_partitions
        print(f'optimizer swapper init, optimizer type: {type(init_optimizer)}')
        assert isinstance(self.optimizer, FP16_Optimizer)

        # fp16 flat groups list
        self.fp16_groups_flat = self.optimizer.fp16_groups_flat

        fp16_partitions = []
        fp32_partitions = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            fp16_partitions.append(
                self.get_partitions(self.fp16_groups_flat[i],
                                    num_partitions))
            #print(f"fp16 partitions, i={i}, partitions={list(map(lambda x: x.numel(), fp16_partitions[i]))}")
            assert len(param_group['params']) == 1
            fp32_partitions.append(
                self.get_partitions(param_group['params'][0],
                                    num_partitions))
            #print(f"fp32 partitions, i={i}, partitions={list(map(lambda x: x.numel(), fp32_partitions[i]))}")
            # swap fp32 partitions out to cpu
            self.move_to_cpu(fp32_partitions[i])
        self.fp16_partitions = fp16_partitions
        self.fp32_partitions = fp32_partitions
        self.num_partitions = num_partitions

    def swap_in_partition(self, partition_idx):
        self.optimizer.fp16_groups = []
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            # set and swap in fp32 partition
            partition = self.fp32_partitions[group_idx][partition_idx]
            self.optimizer.fp32_groups_flat[group_idx] = partition.to('cuda')
            param_group["params"] = [self.optimizer.fp32_groups_flat[group_idx]]
            # move and set fp16 partition and grad
            self.optimizer.fp16_groups.append(
                [self.fp16_partitions[group_idx][partition_idx].to('cuda')])
            self.optimizer.fp16_groups[group_idx][0].grad = self.flat_grad_partitions[
                group_idx][partition_idx].to('cuda')

    def swap_out_partition(self, partition_idx):
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            # move fp32 partition back to cpu
            self.fp32_partitions[group_idx][partition_idx] = param_group["params"][0].to(
                'cpu')
            # since fp16 partitions are not moving devices we don't need to do anything here

    def step(self):
        # partition latest fp16 gradients
        self.flat_grad_partitions = []
        for i, group in enumerate(self.optimizer.fp16_groups):
            # p.grad is None is not properly handled
            self.flat_grad_partitions.append(
                self.get_partitions(_flatten_dense_tensors([p.grad for p in group]),
                                    self.num_partitions))
            self.move_to_cpu(self.flat_grad_partitions[i])
            # clear our fp16 param grads, we don't need them anymore
            for p in group:
                p.grad = None
        # keep ptr to real fp16 groups
        self.real_fp16_groups = self.optimizer.fp16_groups

        print('optimizer swapper step')
        for idx in range(self.num_partitions):
            self.swap_in_partition(idx)
            self.optimizer.step()
            self.swap_out_partition(idx)

        # swap back real fp16 params
        self.optimizer.fp16_groups = self.real_fp16_groups
        self.real_fp16_groups = None

    def backward(self, loss):
        self.optimizer.backward(loss)

    def move_to_cpu(self, tensor_list):
        for tensor in tensor_list:
            tensor.data = tensor.data.cpu()

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
