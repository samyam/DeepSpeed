"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

#########################################
# TENSOR SWAPPING
#########################################
SWAP_TENSOR_FORMAT = '''
"swap_tensor": {
  "folder": "/local_nvme",
  "optimizer": false,
  "optimizer_buffer_count": 10,
  "pipeline_read": false,
  "pipeline_write": false,
  "fp16_params": false,
  "fp16_params_buffer_size": 1e7,
  "fp16_params_buffer_count": 5,
  "max_fp16_params_in_cpu": 100e9
}
'''
SWAP_TENSOR = "swap_tensor"
SWAP_FOLDER = "folder"
SWAP_FOLDER_DEFAULT = None
SWAP_OPTIMIZER = "optimizer"
SWAP_OPTIMIZER_DEFAULT = False
SWAP_OPTIMIZER_BUFFER_COUNT = "optimizer_buffer_count"
SWAP_OPTIMIZER_BUFFER_COUNT_DEFAULT = 10
SWAP_PIPELINE = "swap_pipeline"
SWAP_PIPELINE_READ = "pipeline_read"
SWAP_PIPELINE_READ_DEFAULT = False
SWAP_PIPELINE_WRITE = "pipeline_write"
SWAP_PIPELINE_WRITE_DEFAULT = False
SWAP_FP16_PARAMS = "fp16_params"
SWAP_FP16_PARAMS_DEFAULT = False
SWAP_FP16_PARAMS_BUFFER_SIZE = "fp16_params_buffer_size"
SWAP_FP16_PARAMS_BUFFER_SIZE_DEFAULT = 1e7
SWAP_FP16_PARAMS_BUFFER_COUNT = "fp16_params_buffer_count"
SWAP_FP16_PARAMS_BUFFER_COUNT_DEFAULT = 5
SWAP_MAX_FP16_PARAMS_IN_CPU = "max_fp16_params_in_cpu"
SWAP_MAX_FP16_PARAMS_IN_CPU_DEFAULT = 100e9

#########################################
# AIO
#########################################
AIO_FORMAT = '''
"aio": {
  "block_size": 1048576,
  "queue_depth": 8,
  "thread_count": 1,
  "single_submit": false,
  "overlap_events": true
}
'''
AIO = "aio"
AIO_BLOCK_SIZE = "block_size"
AIO_BLOCK_SIZE_DEFAULT = 1048576
AIO_QUEUE_DEPTH = "queue_depth"
AIO_QUEUE_DEPTH_DEFAULT = 8
AIO_THREAD_COUNT = "thread_count"
AIO_THREAD_COUNT_DEFAULT = 1
AIO_SINGLE_SUBMIT = "single_submit"
AIO_SINGLE_SUBMIT_DEFAULT = False
AIO_OVERLAP_EVENTS = "overlap_events"
AIO_OVERLAP_EVENTS_DEFAULT = True
