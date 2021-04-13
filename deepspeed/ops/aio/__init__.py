'''
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
'''

from ..op_builder import AsyncIOBuilder
aio_op = AsyncIOBuilder().load(verbose=False)
aio_handle = aio_op.aio_handle
aio_read = aio_op.aio_read
aio_write = aio_op.aio_write
deepspeed_memcpy = aio_op.deepspeed_memcpy
