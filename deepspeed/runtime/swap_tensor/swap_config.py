"""
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.swap_tensor.constants import *


def get_swap_tensor_config(param_dict):
    if SWAP_TENSOR in param_dict.keys():
        swap_dict = param_dict[SWAP_TENSOR]
        swap_config = {
            SWAP_FOLDER:
            get_scalar_param(swap_dict,
                             SWAP_FOLDER,
                             SWAP_FOLDER_DEFAULT),
            SWAP_OPTIMIZER:
            get_scalar_param(swap_dict,
                             SWAP_OPTIMIZER,
                             SWAP_OPTIMIZER_DEFAULT),
            SWAP_OPTIMIZER_BUFFER_COUNT:
            get_scalar_param(swap_dict,
                             SWAP_OPTIMIZER_BUFFER_COUNT,
                             SWAP_OPTIMIZER_BUFFER_COUNT_DEFAULT),
            SWAP_PIPELINE_READ:
            get_scalar_param(swap_dict,
                             SWAP_PIPELINE_READ,
                             SWAP_PIPELINE_READ_DEFAULT),
            SWAP_PIPELINE_WRITE:
            get_scalar_param(swap_dict,
                             SWAP_PIPELINE_WRITE,
                             SWAP_PIPELINE_WRITE_DEFAULT),
            SWAP_FP16_PARAMS:
            get_scalar_param(swap_dict,
                             SWAP_FP16_PARAMS,
                             SWAP_FP16_PARAMS_DEFAULT),
            SWAP_FP16_PARAMS_BUFFER_SIZE:
            get_scalar_param(swap_dict,
                             SWAP_FP16_PARAMS_BUFFER_SIZE,
                             SWAP_FP16_PARAMS_BUFFER_SIZE_DEFAULT),
            SWAP_FP16_PARAMS_BUFFER_COUNT:
            get_scalar_param(swap_dict,
                             SWAP_FP16_PARAMS_BUFFER_COUNT,
                             SWAP_FP16_PARAMS_BUFFER_COUNT_DEFAULT),
            SWAP_MAX_FP16_PARAMS_IN_CPU:
            get_scalar_param(swap_dict,
                             SWAP_MAX_FP16_PARAMS_IN_CPU,
                             SWAP_MAX_FP16_PARAMS_IN_CPU_DEFAULT)
        }
        swap_config[SWAP_PIPELINE] = swap_config[SWAP_PIPELINE_READ] or swap_config[
            SWAP_PIPELINE_WRITE]
        return swap_config

    return None
