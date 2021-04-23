

#include <limits>
#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
namespace cg = cooperative_groups;

#define INPUT_TILE 1
#define INPUT_TILE1 1

// Input tile used in the gemm kernel v2
#define INPUT_TILE2 10

#define MAX_REG_SIZE 20

#define WARP_SIZE 32
#define MAX_WARP_NUM 32
#define MAX_BLOCK_SUM 8

#define loop_unroll 4
#define loop_unroll_bits 2

#define inner_loop_unroll 4
#define inner_loop_unroll_bits 2

#define INT8WIDTH 2

#define MAX_QUANTIZE_GROUPING 1024

#define ACC_HALF true

inline __device__ float gelu(const float x)
{
    float y = 0.5 * x * (1.0 + tanhf(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
    return y;
}

__global__ void input_tiled_gemm_kernel_v2(__half* output,
                                           const __half* vals,
                                           const int8_t* weight,
                                           const __half* bias,
                                           unsigned hidden_dim,
                                           unsigned block_reduce,
                                           unsigned input_size,
                                           unsigned output_size,
                                           unsigned outputBlocks,
                                           unsigned blockStride,
                                           float* qscale,
                                           unsigned groups,
                                           __half* block_sums,
                                           unsigned merge_count = 1,
                                           unsigned quantization_stride = 1,
                                           bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* output_cast =
        reinterpret_cast<__half2*>(((gridDim.x == outputBlocks) ? output : block_sums));
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const int16_t* weight_cast = reinterpret_cast<const int16_t*>(weight);
    output_cast += ((blockIdx.x / outputBlocks) * (output_size));
    weight_cast += ((blockIdx.x / outputBlocks) * blockStride);
    vals_cast += (unsigned)(blockIdx.x / outputBlocks) * (hidden_dim >> 1);

    // reading all the quantization scale into a small shared buffer
    __shared__ __half shared_quantize_scale[MAX_QUANTIZE_GROUPING];

    int merge_hidden = hidden_dim >> merge_count;

    if (threadIdx.x < (groups << merge_count))
        shared_quantize_scale[threadIdx.x] = __float2half(qscale[threadIdx.x]);
    __syncthreads();

    for (int j = 0; j < input_size; j += (INPUT_TILE2)) {
        __half2 sum[INPUT_TILE2];
#pragma unroll
        for (int t = 0; t < INPUT_TILE2; t++) sum[t] = __float2half2_rn(0.f);

        {
            int wid = gid << 2;
            weight_cast += (wid * output_size + (blockIdx.x % outputBlocks) * WARP_SIZE + lane);

            while (wid < hidden_dim) {
                // updating the quantization scale
                __half2 qscale_data;
                {
                    auto tmp = shared_quantize_scale[0];
                    qscale_data = __halves2half2(tmp, tmp);
                    if (groups > 1) {
                        unsigned index = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
                        unsigned merge_index = wid / merge_hidden;
                        index = (wid - merge_index * merge_hidden) + (index << 1) * merge_hidden;
                        qscale_data = __halves2half2(
                            shared_quantize_scale[((index / quantization_stride) << merge_count) +
                                                  merge_index],
                            shared_quantize_scale[(((index + merge_hidden) / quantization_stride)
                                                   << merge_count) +
                                                  merge_index]);
                    }
                }
                // Read the input
                __shared__ __half2 vals_h[(loop_unroll >> 1) * INPUT_TILE2 * MAX_WARP_NUM];
                {
                    // we read (loop_unroll >> 2) half-2 values per lane, and for 2 times of the
                    // INPUT_TILE this makes more threads engaged in reading data from shared memory
                    // into registers!
                    if (lane < (INPUT_TILE2 << 1)) {
                        if (((lane >> 1) + j) < input_size) {
                            // here, we consider loop_unroll is always higher that 4!
                            unsigned int inp_id = ((lane % 2) << (loop_unroll_bits - 2));

                            unsigned int offset =
                                (j + (lane >> 1)) * (block_reduce * (hidden_dim >> 1)) + inp_id;
#pragma unroll
                            for (int li = 0; li < (loop_unroll >> 2); li++) {
                                vals_h[li + inp_id + (((lane >> 1) << (loop_unroll_bits - 1))) +
                                       (gid << (loop_unroll_bits - 1)) * INPUT_TILE2] =
                                    vals_cast[offset + (wid >> 1) + li];
                            }
                        }
                    }
                    g.sync();
                }
                int col_index = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
                if (col_index < output_size) {
                    __half2 weight_h[loop_unroll];
                    {
                        int16_t weight_q[loop_unroll];
#pragma unroll
                        for (int k = 0; k < loop_unroll; k++)
                            if ((k + wid) < hidden_dim) weight_q[k] = weight_cast[k * output_size];

#pragma unroll
                        for (int k = 0; k < loop_unroll; k++) {
                            int8_t* weight_8 = reinterpret_cast<int8_t*>(&weight_q[k]);
                            weight_h[k] = __halves2half2(__float2half((float)weight_8[0]),
                                                         __float2half((float)weight_8[1])) *
                                          qscale_data;
                        }
                    }
                    // matrix-matrix multiply
#pragma unroll
                    for (int t = 0; t < INPUT_TILE2; t++) {
                        if ((t + j) < input_size) {
#pragma unroll
                            for (int li = 0; li < loop_unroll; li++) {
                                __half* val_h = reinterpret_cast<__half*>(
                                    &vals_h[(t << (loop_unroll_bits - 1)) + (li >> 1) +
                                            (gid << (loop_unroll_bits - 1)) * INPUT_TILE2]);
                                auto mul =
                                    weight_h[li] * __halves2half2(val_h[li % 2], val_h[li % 2]);
                                if (ACC_HALF)
                                    sum[t] += mul;
                                else {
                                    float2 mul_f = __half22float2(mul);
                                    float2 sum_f = __half22float2(sum[t]);
                                    sum_f.x += mul_f.x;
                                    sum_f.y += mul_f.y;
                                    sum[t] = __float22half2_rn(sum_f);
                                }
                            }
                        }
                    }
                }
                wid += (warp_num << loop_unroll_bits);
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }
        }
        const __half2* bias_cast;
        if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
        __shared__ __half2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];

        for (int t = 0; t < INPUT_TILE2; t += 2) {
            if ((t + j) < input_size) {
                partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1)] = sum[t];
                partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1) + 1] = sum[t + 1];
                b.sync();

                if (ACC_HALF) {
                    sum[t] = partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1)];
                    sum[t + 1] = partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1) + 1];
#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        float temp[2];
                        float* sum_f[2];
                        __half2* sum_h[2];
                        sum_f[0] = reinterpret_cast<float*>(&sum[t]);
                        sum_f[1] = reinterpret_cast<float*>(&sum[t + 1]);
                        temp[0] = g.shfl_xor(*sum_f[0], i);
                        temp[1] = g.shfl_xor(*sum_f[1], i);
                        sum_h[0] = reinterpret_cast<__half2*>(&temp[0]);
                        sum_h[1] = reinterpret_cast<__half2*>(&temp[1]);
                        sum[t] += *sum_h[0];
                        sum[t + 1] += *sum_h[1];
                    }
                    if (lane == 0) {
                        partial_result[(gid << 1)] = sum[t];
                        partial_result[(gid << 1) + 1] = sum[t + 1];
                    }
                } else {
                    float2 sum_f[2];
                    sum_f[0] =
                        __half22float2(partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1)]);
                    sum_f[1] = __half22float2(
                        partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1) + 1]);

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
                        sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
                        sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
                        sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
                    }
                    if (lane == 0) {
                        partial_result[(gid << 1)] = __float22half2_rn(sum_f[0]);
                        partial_result[(gid << 1) + 1] = __float22half2_rn(sum_f[1]);
                    }
                }
                b.sync();

                if (gid == (t >> 1)) {
                    sum[0] = partial_result[(lane << 1)];
                    sum[1] = partial_result[(lane << 1) + 1];
                }
            }
        }

        if ((gid << 1) < INPUT_TILE2 && ((gid << 1) + j) < input_size) {
            int col_index = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
            if (col_index < output_size) {
                if (bias && blockIdx.x < outputBlocks) {
                    __half2 bias_h = bias_cast[col_index];
                    float2 bias_f = __half22float2(bias_h);
                    float2 sum_f[2];
                    sum_f[0] = __half22float2(sum[0]);
                    sum_f[1] = __half22float2(sum[1]);
                    sum_f[0].x += bias_f.x;
                    sum_f[0].y += bias_f.y;
                    sum_f[1].x += bias_f.x;
                    sum_f[1].y += bias_f.y;
                    if (add_gelu && gridDim.x == outputBlocks) {
                        sum_f[0].x = gelu(sum_f[0].x);
                        sum_f[1].x = gelu(sum_f[1].x);
                        sum_f[0].y = gelu(sum_f[0].y);
                        sum_f[1].y = gelu(sum_f[1].y);
                    }
                    sum[0] = __float22half2_rn(sum_f[0]);
                    sum[1] = __float22half2_rn(sum_f[1]);
                }
                output_cast[col_index + (j + (gid << 1)) * (block_reduce * output_size)] = (sum[0]);
                if ((input_size - ((gid << 1) + j)) > 1)
                    output_cast[col_index + (j + (gid << 1) + 1) * (block_reduce * output_size)] =
                        (sum[1]);
            }
        }
        weight_cast = reinterpret_cast<const int16_t*>(weight);
        weight_cast += ((blockIdx.x / outputBlocks) * blockStride);
    }
}
__global__ void input_tiled_gemm_kernel_v2(float* output,
                                           const float* vals,
                                           const float* weight,
                                           const float* bias,
                                           float* block_sums,
                                           int hidden_dim,
                                           int block_reduce,
                                           int input_size,
                                           int output_size,
                                           int outputBlocks,
                                           int blockStride,
                                           bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;

    int warp_num = blockDim.x >> 5;

    float2* output_cast =
        reinterpret_cast<float2*>(((gridDim.x == outputBlocks) ? output : block_sums));
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);
    output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
    int hidden_half = hidden_dim >> 1;
    weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
    vals_cast += (unsigned)(blockIdx.x / outputBlocks) * hidden_half;
    for (int j = 0; j < input_size; j += (INPUT_TILE2)) {
        float2 sum[INPUT_TILE2];
#pragma unroll
        for (int t = 0; t < (INPUT_TILE2); t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        {
            int wid = gid << 1;
            int offset = wid * output_size;

            while (wid < hidden_dim) {
                float2 val_data[INPUT_TILE2];
                {
                    for (int t = 0; t < INPUT_TILE2; t++) {
                        if ((t + j) < input_size) {
                            val_data[t] =
                                vals_cast[(j + t) * (hidden_half * block_reduce) + (wid >> 1)];
                        }
                    }
                }

                int row = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    float2 weight[2];
                    weight[0] = weight_cast[offset1];
                    weight[1] = weight_cast[output_size + offset1];

                    for (int t = 0; t < INPUT_TILE2; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[2];
                            mul[0].x = val_data[t].x * weight[0].x;
                            mul[0].y = val_data[t].x * weight[0].y;
                            mul[1].x = val_data[t].y * weight[1].x;
                            mul[1].y = val_data[t].y * weight[1].y;

                            sum[t].x += mul[0].x + mul[1].x;
                            sum[t].y += mul[0].y + mul[1].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 2;
                offset += (output_size * warp_num * 2);
            }
        }
        {
            const float2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
            __shared__ float2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < (INPUT_TILE2); t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid * (WARP_SIZE + 1) + lane] = sum_g;
                    __syncthreads();

                    sum_g = partial_result[lane * (WARP_SIZE + 1) + gid];
                    __syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[gid] = sum_g; }

                    __syncthreads();
                    sum[t] = partial_result[lane];
                }
            }
            if (gid < INPUT_TILE2 && ((gid + j) < input_size)) {
                int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
                if (col < output_size) {
                    if (bias && blockIdx.x < outputBlocks) {
                        float2 bias_f = bias_cast[col];
                        sum[gid].x += bias_f.x;
                        sum[gid].y += bias_f.y;
                        if (add_gelu && gridDim.x == outputBlocks) {
                            sum[gid].x = gelu(sum[gid].x);
                            sum[gid].y = gelu(sum[gid].y);
                        }
                    }
                    output_cast[col + (j + gid) * (output_size * block_reduce)] = sum[gid];
                }
            }
        }
    }
}

__global__ void input_tiled_gemm_kernel_v2(__half* output,
                                           const __half* vals,
                                           const __half* weight,
                                           const __half* bias,
                                           __half* block_sums,
                                           unsigned int hidden_dim,
                                           unsigned int block_reduce,
                                           unsigned int input_size,
                                           unsigned int output_size,
                                           unsigned int outputBlocks,
                                           unsigned int blockStride,
                                           bool add_gelu = false)
{
#if __CUDA_ARCH__ >= 700
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;

    int warp_num = blockDim.x >> 5;

    __half2* output_cast =
        reinterpret_cast<__half2*>(((gridDim.x == outputBlocks) ? output : block_sums));
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);
    output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
    int hidden_half = hidden_dim >> 1;
    weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
    vals_cast += (unsigned)(blockIdx.x / outputBlocks) * hidden_half;

    for (int j = 0; j < input_size; j += (INPUT_TILE2)) {
        __half2 sum[INPUT_TILE2];
#pragma unroll
        for (int t = 0; t < INPUT_TILE2; t++) { sum[t] = __float2half2_rn(0.f); }

        {
            int wid = gid << loop_unroll_bits;
            weight_cast += wid * output_size + (blockIdx.x % outputBlocks) * WARP_SIZE + lane;

            while (wid < hidden_dim) {
                __shared__ __half2 vals_h[(loop_unroll >> 1) * INPUT_TILE2 * MAX_WARP_NUM];
                {
                    // we read (loop_unroll >> 2) half-2 values per lane, and for 2 times of the
                    // INPUT_TILE this makes more threads engaged in reading data from shared memory
                    // into registers!
                    if (lane < (INPUT_TILE2 << 1)) {
                        if (((lane >> 1) + j) < input_size) {
                            // here, we consider loop_unroll is always higher that 4!
                            unsigned int inp_id = ((lane % 2) << (loop_unroll_bits - 2));

                            unsigned int offset =
                                (j + (lane >> 1)) * (block_reduce * (hidden_dim >> 1)) + inp_id;
#pragma unroll
                            for (int li = 0; li < (loop_unroll >> 2); li++) {
                                vals_h[li + inp_id + (((lane >> 1) << (loop_unroll_bits - 1))) +
                                       (gid << (loop_unroll_bits - 1)) * INPUT_TILE2] =
                                    vals_cast[offset + (wid >> 1) + li];
                            }
                        }
                    }
                    g.sync();
                }

                int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;

                if (col < output_size) {
                    __half2 weight_h[loop_unroll];
#pragma unroll
                    for (int k = 0; k < loop_unroll; k++)
                        weight_h[k] = weight_cast[output_size * k];
#pragma unroll
                    for (int t = 0; t < INPUT_TILE2; t++) {
                        float2 sum_f;
                        if (!ACC_HALF) sum_f = __half22float2(sum[t]);
#pragma unroll
                        for (int li = 0; li < (loop_unroll >> 1); li++) {
                            __half* inp_data = reinterpret_cast<__half*>(
                                &vals_h[(t << (loop_unroll_bits - 1)) + li +
                                        (gid << (loop_unroll_bits - 1)) * INPUT_TILE2]);
#pragma unroll
                            for (int k = 0; k < 2; k++) {
                                if (ACC_HALF)
                                    sum[t] += __halves2half2(inp_data[k], inp_data[k]) *
                                              weight_h[(li << 1) + k];
                                else {
                                    float2 weight_f =
                                        __half22float2(__halves2half2(inp_data[k], inp_data[k]) *
                                                       weight_h[(li << 1) + k]);
                                    sum_f.x += weight_f.x;
                                    sum_f.y += weight_f.y;
                                }
                            }
                        }
                        if (!ACC_HALF) sum[t] = __float22half2_rn(sum_f);
                    }
                }
                wid += warp_num << loop_unroll_bits;
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }
        }
        {
            const __half2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
            __shared__ __half2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];

            for (int t = 0; t < INPUT_TILE2; t += 2) {
                if ((t + j) < input_size) {
                    partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1)] = sum[t];
                    partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1) + 1] = sum[t + 1];
                    b.sync();

                    float2 sum_f[2];
                    sum_f[0] =
                        __half22float2(partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1)]);
                    sum_f[1] = __half22float2(
                        partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1) + 1]);

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
                        sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
                        sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
                        sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
                    }

                    if (lane == 0) {
                        partial_result[(gid << 1)] = __float22half2_rn(sum_f[0]);
                        partial_result[(gid << 1) + 1] = __float22half2_rn(sum_f[1]);
                    }
                    b.sync();

                    if (gid == (t >> 1)) {
                        sum[t] = partial_result[(lane << 1)];
                        sum[t + 1] = partial_result[(lane << 1) + 1];
                    }
                }
            }

            if ((gid << 1) < INPUT_TILE2 && ((gid << 1) + j) < input_size) {
                int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
                if (col < output_size) {
                    if (bias && blockIdx.x < outputBlocks) {
                        __half2 bias_h = bias_cast[col];
                        float2 bias_f = __half22float2(bias_h);
                        float2 sum_f[2];
                        sum_f[0] = __half22float2(sum[(gid << 1)]);
                        sum_f[1] = __half22float2(sum[(gid << 1) + 1]);
                        sum_f[0].x += bias_f.x;
                        sum_f[0].y += bias_f.y;
                        sum_f[1].x += bias_f.x;
                        sum_f[1].y += bias_f.y;
                        if (add_gelu && gridDim.x == outputBlocks) {
                            sum_f[0].x = gelu(sum_f[0].x);
                            sum_f[0].y = gelu(sum_f[0].y);
                            sum_f[1].x = gelu(sum_f[1].x);
                            sum_f[1].y = gelu(sum_f[1].y);
                        }
                        sum[(gid << 1)] = __float22half2_rn(sum_f[0]);
                        sum[(gid << 1) + 1] = __float22half2_rn(sum_f[0]);
                    }
                    output_cast[col + (j + (gid << 1)) * (block_reduce * output_size)] =
                        (sum[(gid << 1)]);
                    if (((gid << 1) + j + 1) < input_size)
                        output_cast[col + (j + (gid << 1) + 1) * (block_reduce * output_size)] =
                            (sum[(gid << 1) + 1]);
                }
            }
        }
        weight_cast = reinterpret_cast<const __half2*>(weight);
        weight_cast += ((blockIdx.x / outputBlocks) * blockStride);
    }
#endif
}

__global__ void block_reduce_kernel(float* output,
                                    float* block_sums,
                                    int batch,
                                    int output_size,
                                    bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    unsigned total_count = batch * output_size;

    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;
    unsigned int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    float2* block_sums_cast = reinterpret_cast<float2*>(block_sums);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    block_sums_cast += gid * output_size;

    if (col_index < total_count) {
        __shared__ float2 data_shared[MAX_WARP_NUM * (WARP_SIZE + 1)];

        data_shared[gid * (WARP_SIZE) + lane] =
            block_sums_cast[(col_index / output_size) * (warp_num * output_size) +
                            col_index % output_size];

        b.sync();

        float2 data = data_shared[(lane % warp_num) * WARP_SIZE + gid * (WARP_SIZE / warp_num) +
                                  (lane / warp_num)];

        b.sync();
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            data.x += g.shfl_down(data.x, i);
            data.y += g.shfl_down(data.y, i);
        }

        if ((lane % warp_num) == 0) {
            if (add_gelu) {
                data.x = gelu(data.x);
                data.y = gelu(data.y);
            }
            data_shared[gid * (WARP_SIZE / warp_num) + (lane / warp_num)] = (data);
        }

        b.sync();

        if (gid == 0) output_cast[col_index] = data_shared[lane];
    }
}
__global__ void block_reduce_kernel(__half* output,
                                    __half* block_sums,
                                    unsigned batch,
                                    unsigned int output_size,
                                    bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    unsigned total_count = batch * output_size;
    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;
    unsigned int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    __half2* block_sums_cast = reinterpret_cast<__half2*>(block_sums);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    block_sums_cast += gid * output_size;

    if (col_index < total_count) {
        __shared__ __half2 data_shared[MAX_WARP_NUM * (WARP_SIZE + 1)];

        data_shared[gid * (WARP_SIZE) + lane] =
            block_sums_cast[(col_index / output_size) * (warp_num * output_size) +
                            col_index % output_size];

        b.sync();

        float2 data = __half22float2(data_shared[(lane % warp_num) * WARP_SIZE +
                                                 gid * (WARP_SIZE / warp_num) + (lane / warp_num)]);

        b.sync();
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            data.x += g.shfl_down(data.x, i);
            data.y += g.shfl_down(data.y, i);
        }

        if ((lane % warp_num) == 0) {
            if (add_gelu) {
                data.x = gelu(data.x);
                data.y = gelu(data.y);
            }
            data_shared[gid * (WARP_SIZE / warp_num) + (lane / warp_num)] = __float22half2_rn(data);
        }

        b.sync();

        if (gid == 0) output_cast[col_index] = data_shared[lane];
    }
}
template <typename T>
void launch_input_tiled_gemm_kernel_v2(T* output,
                                       const T* vals,
                                       const int8_t* weight,
                                       const T* bias,
                                       unsigned int hidden_dim,
                                       unsigned int input_size,
                                       unsigned int output_size,
                                       float* scale,
                                       unsigned int groups,
                                       unsigned int merge_count,
                                       T* block_sums,
                                       bool add_gelu,
                                       cudaStream_t stream)
{
    output_size /= 2;
    int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

    int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
    int br2 = (int)log2(block_reduce);
    block_reduce = (int)pow(2.0, (float)br2);

    constexpr int threads = 1024;
    int blockStride = (output_size * hidden_dim) / block_reduce;
    dim3 grid_dim(outputBlocks * block_reduce);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel_v2<<<grid_dim, block_dim, 0, stream>>>(
        output,
        vals,
        weight,
        bias,
        hidden_dim / block_reduce,
        block_reduce,
        input_size,
        output_size,
        outputBlocks,
        blockStride,
        scale,
        groups,
        block_sums,
        merge_count,
        ((hidden_dim >> merge_count) * (output_size << 1)) / groups,
        add_gelu);
    if (block_reduce > 1) {
        dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
        dim3 blocks(block_reduce * WARP_SIZE);
        block_reduce_kernel<<<grids, blocks, 0, stream>>>(
            output, block_sums, input_size, (output_size), add_gelu);
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel_v2(T* output,
                                       const T* vals,
                                       const T* weight,
                                       const T* bias,
                                       T* block_sums,
                                       unsigned int hidden_dim,
                                       unsigned int input_size,
                                       unsigned int output_size,
                                       bool add_gelu,
                                       cudaStream_t stream)
{
    output_size /= 2;
    int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

    int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
    int br2 = (int)log2(block_reduce);
    block_reduce = (int)pow(2.0, (float)br2);

    constexpr int threads = 1024;
    int blockStride = (output_size * hidden_dim) / block_reduce;

    dim3 grid_dim(outputBlocks * block_reduce);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel_v2<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                   vals,
                                                                   weight,
                                                                   bias,
                                                                   block_sums,
                                                                   hidden_dim / block_reduce,
                                                                   block_reduce,
                                                                   input_size,
                                                                   output_size,
                                                                   outputBlocks,
                                                                   blockStride,
                                                                   add_gelu);
    if (block_reduce > 1) {
        dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
        dim3 blocks(block_reduce * WARP_SIZE);
        block_reduce_kernel<<<grids, blocks, 0, stream>>>(
            output, block_sums, input_size, (output_size), add_gelu);
    }
}
template void launch_input_tiled_gemm_kernel_v2(__half* output,
                                                const __half* vals,
                                                const __half* weight,
                                                const __half* bias,
                                                __half* block_sums,
                                                unsigned int hidden_dim,
                                                unsigned int input_size,
                                                unsigned int output_size,
                                                bool add_gelu,
                                                cudaStream_t stream);

template void launch_input_tiled_gemm_kernel_v2(float* output,
                                                const float* vals,
                                                const float* weight,
                                                const float* bias,
                                                float* block_sums,
                                                unsigned int hidden_dim,
                                                unsigned int input_size,
                                                unsigned int output_size,
                                                bool add_gelu,
                                                cudaStream_t stream);

template void launch_input_tiled_gemm_kernel_v2(__half* output,
                                                const __half* vals,
                                                const int8_t* weight,
                                                const __half* bias,
                                                unsigned int hidden_dim,
                                                unsigned int input_size,
                                                unsigned int output_size,
                                                float* scale,
                                                unsigned int groups,
                                                unsigned int merge_count,
                                                __half* block_sums,
                                                bool add_gelu,
                                                cudaStream_t stream);

__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const int8_t* weight,
                                        const __half* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        float* qscale,
                                        int groups,
                                        int merge_count = 1)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const int16_t* weight_cast = reinterpret_cast<const int16_t*>(weight);

    int hidden_half = hidden_dim >> 1;
    int merge_hidden = hidden_dim >> merge_count;
    int quantization_stride = (merge_hidden * (output_size << 1)) / groups;

    // reading all the quantization scale into a small shared buffer
    __shared__ __half shared_quantize_scale[MAX_QUANTIZE_GROUPING];

    if (threadIdx.x < (groups << merge_count))
        shared_quantize_scale[threadIdx.x] = __float2half(qscale[threadIdx.x]);
    __syncthreads();

    int col_index = blockIdx.x * WARP_SIZE + lane;

    for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
        __half2 sum[INPUT_TILE1];
#pragma unroll
        for (int t = 0; t < INPUT_TILE1; t++) sum[t] = __float2half2_rn(0.f);

        {
            int wid = gid << 2;
            weight_cast += (wid * output_size + col_index);

            while (wid < hidden_dim) {
                // updating the quantization scale

                __half2 qscale_data;
                {
                    auto tmp = shared_quantize_scale[0];
                    qscale_data = __halves2half2(tmp, tmp);
                    if (groups > 1) {
                        unsigned index;
                        unsigned merge_index = wid / merge_hidden;
                        index =
                            (wid - merge_index * merge_hidden) + (col_index << 1) * merge_hidden;
                        qscale_data = __halves2half2(
                            shared_quantize_scale[((index / quantization_stride) << merge_count) +
                                                  merge_index],
                            shared_quantize_scale[(((index + merge_hidden) / quantization_stride)
                                                   << merge_count) +
                                                  merge_index]);
                    }
                }
                __half2 vals_f[INPUT_TILE1 * loop_unroll];

#pragma unroll
                for (int t = 0; t < INPUT_TILE1; t++) {
                    __half2 val_h[loop_unroll >> 1];
                    val_h[0] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
                    val_h[1] = vals_cast[(j + t) * hidden_half + (wid >> 1) + 1];

                    __half* inp_data[2];
                    inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                    inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);
                    vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                    vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                    vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                    vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                }

                if (col_index < output_size) {
                    int16_t weight_q[loop_unroll];
#pragma unroll
                    for (int k = 0; k < loop_unroll; k++)
                        weight_q[k] = weight_cast[k * output_size];

#pragma unroll
                    for (int t = 0; t < INPUT_TILE1; t++) {
#pragma unroll
                        for (int li = 0; li < loop_unroll; li++) {
                            float2 weight_f;
                            int8_t* weight_8 = reinterpret_cast<int8_t*>(&weight_q[li]);
                            weight_f.x = (float)weight_8[0];
                            weight_f.y = (float)weight_8[1];
                            auto mul =
                                __float22half2_rn(weight_f) * qscale_data * vals_f[(t << 2) + li];
                            if (ACC_HALF)
                                sum[t] += mul;
                            else {
                                float2 mul_f = __half22float2(mul);
                                float2 sum_f = __half22float2(sum[t]);
                                sum_f.x += mul_f.x;
                                sum_f.y += mul_f.y;
                                sum[t] = __float22half2_rn(sum_f);
                            }
                        }
                    }
                }
                wid += (warp_num << loop_unroll_bits);
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }
        }
        {
            const __half2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
            __shared__ __half2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < INPUT_TILE1; t++) {
                partial_result[gid * (WARP_SIZE + 1) + lane] = sum[t];
                __syncthreads();

                sum[t] = partial_result[lane * (WARP_SIZE + 1) + gid];
                if (ACC_HALF) {
#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        float* sum_f = reinterpret_cast<float*>(&sum[t]);
                        float temp = g.shfl_xor(*sum_f, i);
                        __half2* sum_h = reinterpret_cast<__half2*>(&temp);
                        sum[t] += *sum_h;
                    }
                } else {
                    float2 sum_f = __half22float2(sum[t]);
#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_f.x += g.shfl_xor(sum_f.x, i);
                        sum_f.y += g.shfl_xor(sum_f.y, i);
                    }
                    sum[t] = __float22half2_rn(sum_f);
                }

                if (lane == 0) { partial_result[gid] = sum[t]; }

                __syncthreads();

                if (gid == t) sum[0] = partial_result[lane];
            }
            if (gid < (INPUT_TILE1) && (gid + j) < input_size && col_index < output_size) {
                if (bias) {
                    float2 bias_f = __half22float2(bias_cast[col_index]);
                    float2 sum_f = __half22float2(sum[0]);
                    sum_f.x += bias_f.x;
                    sum_f.y += bias_f.y;
                    sum[0] = __float22half2_rn(sum_f);
                }
                output_cast[col_index + (j + gid) * output_size] = sum[0];
            }
        }
        weight_cast = reinterpret_cast<const int16_t*>(weight);
    }
}
__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const __half* weight,
                                        const __half* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
        __half2 sum[INPUT_TILE1];
#pragma unroll
        for (int t = 0; t < (INPUT_TILE1); t++) { sum[t] = __float2half2_rn(0.f); }

        {
            int wid = gid << loop_unroll_bits;
            weight_cast += (wid * output_size + col_index);

            while (wid < hidden_dim) {
                __half2 vals_f[loop_unroll * (INPUT_TILE1)];
                {
                    for (int t = 0; t < (INPUT_TILE1); t++) {
                        if ((t + j) < input_size) {
                            __half2 val_h[2];
                            val_h[0] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
                            val_h[1] = vals_cast[(j + t) * hidden_half + (wid >> 1) + 1];

                            __half* inp_data[2];
                            inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                            inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                            vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                            vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                            vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                            vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                        }
                    }
                }

                if (col_index < output_size) {
                    __half2 weight_h[loop_unroll];
#pragma unroll
                    for (int k = 0; k < loop_unroll; k++) {
                        if ((k + wid) < hidden_dim)
                            weight_h[k] = weight_cast[k * output_size];
                        else
                            weight_h[k] = __float2half2_rn(0.f);
                    }

#pragma unroll
                    for (int k = 0; k < (loop_unroll >> inner_loop_unroll_bits); k++) {
#pragma unroll
                        for (int t = 0; t < (INPUT_TILE1); t++) {
                            if ((t + j) < input_size) {
#pragma unroll
                                for (int li = 0; li < inner_loop_unroll; li++) {
                                    weight_h[0] = (vals_f[(t << 2) + li] * weight_h[li]);
                                    if (ACC_HALF)
                                        sum[t] += weight_h[0];
                                    else {
                                        float2 weight_f = __half22float2(weight_h[0]);
                                        float2 sum_f = __half22float2(sum[t]);
                                        sum_f.x += weight_f.x;
                                        sum_f.y += weight_f.y;
                                        sum[t] = __float22half2_rn(sum_f);
                                    }
                                }
                            }
                        }
                    }
                }
                wid += (warp_num << loop_unroll_bits);
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }
        }
        {
            const __half2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
            __shared__ __half2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < (INPUT_TILE1); t++) {
                if ((t + j) < input_size) {
                    __half2 sum_g = sum[t];
                    partial_result[gid * (WARP_SIZE + 1) + lane] = sum[t];

                    b.sync();
                    float2 sum_f;
                    sum_f = __half22float2(partial_result[lane * (WARP_SIZE + 1) + gid]);

                    b.sync();
#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_f.x += g.shfl_xor(sum_f.x, i);
                        sum_f.y += g.shfl_xor(sum_f.y, i);
                    }

                    if (lane == 0) { partial_result[gid] = __float22half2_rn(sum_f); }

                    b.sync();

                    if (gid == 0) {
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            sum_g = partial_result[lane];
                            if (bias) {
                                float2 bias_f = __half22float2(bias_cast[col]);
                                sum_f = __half22float2(sum_g);
                                sum_f.x += bias_f.x;
                                sum_f.y += bias_f.y;
                                sum_g = __float22half2_rn(sum_f);
                            }
                            output_cast[col + (j + t) * output_size] = (sum_g);
                        }
                    }
                }
            }
        }
        weight_cast = reinterpret_cast<const __half2*>(weight);
    }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output,
                                        const float* vals,
                                        const float* weight,
                                        const float* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
        float2 sum[INPUT_TILE1];
#pragma unroll
        for (int t = 0; t < (INPUT_TILE1); t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        {
            int wid = gid << 1;
            int offset = wid * output_size;

            while (wid < hidden_dim) {
                float2 val_data[INPUT_TILE1];
                {
                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            val_data[t] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
                        }
                    }
                }

                int row = blockIdx.x * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    float2 weight[2];
                    weight[0] = weight_cast[offset1];
                    weight[1] = weight_cast[output_size + offset1];

                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[2];
                            mul[0].x = val_data[t].x * weight[0].x;
                            mul[0].y = val_data[t].x * weight[0].y;
                            mul[1].x = val_data[t].y * weight[1].x;
                            mul[1].y = val_data[t].y * weight[1].y;

                            sum[t].x += mul[0].x + mul[1].x;
                            sum[t].y += mul[0].y + mul[1].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 2;
                offset += (output_size * warp_num * 2);
            }
        }
        {
            const float2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
            __shared__ float2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < (INPUT_TILE1); t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid * (WARP_SIZE + 1) + lane] = sum_g;
                    __syncthreads();

                    sum_g = partial_result[lane * (WARP_SIZE + 1) + gid];
                    __syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[gid] = sum_g; }

                    __syncthreads();

                    if (gid == 0) {
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            sum_g = partial_result[lane];
                            if (bias) {
                                float2 bias_f = bias_cast[col];
                                sum_g.x += bias_f.x;
                                sum_g.y += bias_f.y;
                            }
                            output_cast[col + (j + t) * output_size] = sum_g;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, weight, bias, hidden_dim, input_size, output_size);
}

template void launch_input_tiled_gemm_kernel(float* output,
                                             const float* vals,
                                             const float* weight,
                                             const float* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const __half* weight,
                                             const __half* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const int8_t* weight,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    float* scale,
                                    int groups,
                                    int merge_count,
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                vals,
                                                                weight,
                                                                bias,
                                                                hidden_dim,
                                                                input_size,
                                                                output_size,
                                                                scale,
                                                                groups,
                                                                merge_count);
}

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const int8_t* weight,
                                             const __half* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             float* scale,
                                             int groups,
                                             int merge_count,
                                             cudaStream_t stream);

__global__ void tiled_gemm_kernel_gelu(__half* output,
                                       const __half* vals,
                                       const __half* weight,
                                       const __half* bias,
                                       int hidden_dim,
                                       int input_size,
                                       int output_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half2 inp_reg[8];

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

    int input_tile = (input_size < INPUT_TILE ? input_size : INPUT_TILE);
    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += input_tile) {
        __shared__ __half2 input_shared[9000];

        {
            int k = 0;
            int input_id = id;
            while (input_id < hidden_half) {
                inp_reg[k] = vals_cast[j * hidden_half + input_id];
                float2 inp_f = __half22float2(inp_reg[k]);
                inp_f.x = gelu(inp_f.x);
                inp_f.y = gelu(inp_f.y);
                inp_reg[k] = __float22half2_rn(inp_f);
                input_shared[input_id] = inp_reg[k++];
                input_id += blockDim.x;
            }
            b.sync();
        }

        int wid = gid << 2;
        int offset = wid * output_size;
        float2 sum;
        sum.x = 0;
        sum.y = 0;

        while (wid < hidden_dim) {
            __half2 vals_f[4];

            {
                __half2 val_h[2];
                val_h[0] = input_shared[(wid >> 1)];
                val_h[1] = input_shared[(wid >> 1) + 1];

                __half* inp_data[2];
                inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                vals_f[0] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                vals_f[1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                vals_f[2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                vals_f[3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                __half2 weight_h[4];
                weight_h[0] = weight_cast[offset1];
                weight_h[1] = weight_cast[output_size + offset1];
                weight_h[2] = weight_cast[(output_size << 1) + offset1];
                weight_h[3] = weight_cast[((output_size << 1) + output_size) + offset1];

                {
                    float2 mul[4];
                    mul[0] = __half22float2(vals_f[0] * weight_h[0]);
                    mul[1] = __half22float2(vals_f[1] * weight_h[1]);
                    mul[2] = __half22float2(vals_f[2] * weight_h[2]);
                    mul[3] = __half22float2(vals_f[3] * weight_h[3]);

                    sum.x += mul[0].x + mul[1].x + mul[2].x + mul[3].x;
                    sum.y += mul[0].y + mul[1].y + mul[2].y + mul[3].y;
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 4;
            offset += (output_size * warp_num * 4);
        }

        __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];

        {
            partial_result[gid][lane] = sum;
            __syncthreads();
            sum = partial_result[lane][gid];
            __syncthreads();
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum.x += g.shfl_xor(sum.x, i);
            sum.y += g.shfl_xor(sum.y, i);
        }

        if (lane == 0) { partial_result[0][gid] = sum; }
        __syncthreads();

        if (gid == 0) {
            sum = partial_result[gid][lane];
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) { output_cast[j * output_size + col] = __float22half2_rn(sum); }
        }
    }
#endif
}

__global__ void tiled_gemm_kernel_gelu(float* output,
                                       const float* vals,
                                       const float* weight,
                                       const float* bias,
                                       int hidden_dim,
                                       int input_size,
                                       int output_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float2 inp_reg[8];

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    int input_tile = (input_size < INPUT_TILE ? input_size : INPUT_TILE);
    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += input_tile) {
        __shared__ float2 input_shared[5000];

        {
            int k = 0;
            int input_id = id;
            while (input_id < hidden_half) {
                inp_reg[k] = vals_cast[j * hidden_half + input_id];
                inp_reg[k].x = gelu(inp_reg[k].x);
                inp_reg[k].y = gelu(inp_reg[k].y);
                input_shared[input_id] = inp_reg[k++];
                input_id += blockDim.x;
            }
            b.sync();
        }

        int wid = gid << 1;
        int offset = wid * output_size;
        float2 sum;
        sum.x = 0;
        sum.y = 0;

        while (wid < hidden_dim) {
            float2 val_data;

            {
                val_data = input_shared[wid >> 1];
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                float2 weight[2];
                weight[0] = weight_cast[offset1];
                weight[1] = weight_cast[output_size + offset1];

                {
                    float2 mul[4];
                    mul[0].x = val_data.x * weight[0].x;
                    mul[0].y = val_data.x * weight[0].y;
                    mul[1].x = val_data.y * weight[1].x;
                    mul[1].y = val_data.y * weight[1].y;

                    sum.x += mul[0].x + mul[1].x;
                    sum.y += mul[0].y + mul[1].y;
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 2;
            offset += (output_size * warp_num * 2);
        }

        __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];

        {
            partial_result[gid][lane] = sum;
            __syncthreads();
            sum = partial_result[lane][gid];
            __syncthreads();
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum.x += g.shfl_xor(sum.x, i);
            sum.y += g.shfl_xor(sum.y, i);
        }

        if (lane == 0) { partial_result[0][gid] = sum; }
        __syncthreads();

        if (gid == 0) {
            sum = partial_result[gid][lane];
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) { output_cast[(j)*output_size + col] = sum; }
        }
    }
}

template <typename T>
void launch_tiled_gemm_kernel_gelu(T* output,
                                   const T* vals,
                                   const T* weight,
                                   const T* bias,
                                   int hidden_dim,
                                   int input_size,
                                   int output_size,
                                   cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, weight, bias, hidden_dim, input_size, output_size);
}

template void launch_tiled_gemm_kernel_gelu(float* output,
                                            const float* vals,
                                            const float* weight,
                                            const float* bias,
                                            int hidden_dim,
                                            int input_size,
                                            int output_size,
                                            cudaStream_t stream);

template void launch_tiled_gemm_kernel_gelu(__half* output,
                                            const __half* vals,
                                            const __half* weight,
                                            const __half* bias,
                                            int hidden_dim,
                                            int input_size,
                                            int output_size,
                                            cudaStream_t stream);

__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const int8_t* weight,
                                        const __half* bias,
                                        const __half* gamma,
                                        const __half* beta,
                                        const float epsilon,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        float* qscale,
                                        int groups)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;
    int quantization_stride = (hidden_dim * (output_size << 1)) / groups;
    int col_index = blockIdx.x * WARP_SIZE + lane;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const int16_t* weight_cast = reinterpret_cast<const int16_t*>(weight);
    // used for quantization scaling factor
    __shared__ __half shared_quantize_scale[MAX_QUANTIZE_GROUPING];
    // reading all the quantization scale into a small shared buffer
    if (threadIdx.x < groups)
        shared_quantize_scale[threadIdx.x] = __float2half(qscale[threadIdx.x]);
    __syncthreads();
    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ __half2 input_shared[9000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                __half2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    sum += inp_f.x + inp_f.y;
                }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    inp_f.x -= mean;
                    inp_f.y -= mean;
                    inp_reg[f] = __float22half2_rn(inp_f);
                    sum += inp_f.x * inp_f.x;
                    sum += inp_f.y * inp_f.y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                __half2 variance_h = __float2half2_rn(sum);
                const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
                const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f] = inp_reg[f] * h2rsqrt(variance_h);
                    inp_reg[f] = inp_reg[f] * gamma_cast[id] + beta_cast[id];
                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        {
            int wid = gid << 2;
            weight_cast += (wid * output_size + col_index);

            __half2 sum[INPUT_TILE];
            for (int t = 0; t < INPUT_TILE; t++) { sum[t] = __float2half2_rn(0.f); }

            while (wid < hidden_dim) {
                // updating the quantization scale

                __half2 qscale_data;
                {
                    auto tmp = shared_quantize_scale[0];
                    qscale_data = __halves2half2(tmp, tmp);
                    if (groups > 1) {
                        unsigned index;
                        index = wid + (col_index << 1) * hidden_dim;
                        qscale_data = __halves2half2(
                            shared_quantize_scale[index / quantization_stride],
                            shared_quantize_scale[(index + hidden_dim) / quantization_stride]);
                    }
                }
                __half2 vals_f[4 * INPUT_TILE];
                for (int t = 0; t < INPUT_TILE; t++) {
                    __half2 val_h[2];
                    val_h[0] = input_shared[t * hidden_half + (wid >> 1)];
                    val_h[1] = input_shared[t * hidden_half + (wid >> 1) + 1];

                    __half* inp_data[2];
                    inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                    inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                    vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                    vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                    vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                    vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                }

                if (col_index < output_size) {
                    int16_t weight_q[loop_unroll];

#pragma unroll
                    for (int k = 0; k < loop_unroll; k++)
                        weight_q[k] = weight_cast[k * output_size];

#pragma unroll
                    for (int t = 0; t < INPUT_TILE; t++) {
#pragma unroll
                        for (int li = 0; li < loop_unroll; li++) {
                            float2 weight_f;
                            int8_t* weight_8 = reinterpret_cast<int8_t*>(&weight_q[li]);
                            weight_f.x = (float)weight_8[0];
                            weight_f.y = (float)weight_8[1];
                            auto mul =
                                __float22half2_rn(weight_f) * qscale_data * vals_f[(t << 2) + li];
                            if (ACC_HALF)
                                sum[t] += mul;
                            else {
                                float2 mul_f = __half22float2(mul);
                                float2 sum_f = __half22float2(sum[t]);
                                sum_f.x += mul_f.x;
                                sum_f.y += mul_f.y;
                                sum[t] = __float22half2_rn(sum_f);
                            }
                        }
                    }
                }
                wid += (warp_num << loop_unroll_bits);
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }

            __shared__ __half2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
            const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    partial_result[gid][lane] = sum[t];
                    __syncthreads();
                    sum[t] = partial_result[lane][gid];

                    if (ACC_HALF) {
#pragma unroll
                        for (int i = 1; i < WARP_SIZE; i *= 2) {
                            float* sum_f = reinterpret_cast<float*>(&sum[t]);
                            float temp = g.shfl_xor(*sum_f, i);
                            __half2* sum_h = reinterpret_cast<__half2*>(&temp);
                            sum[t] += *sum_h;
                        }
                    } else {
                        float2 sum_g = __half22float2(sum[t]);
#pragma unroll
                        for (int i = 1; i < WARP_SIZE; i *= 2) {
                            sum_g.x += g.shfl_xor(sum_g.x, i);
                            sum_g.y += g.shfl_xor(sum_g.y, i);
                        }
                        sum[t] = __float22half2_rn(sum_g);
                    }

                    if (lane == 0) { partial_result[0][gid] = sum[0]; }
                    __syncthreads();

                    if (gid == 0) {
                        sum[0] = partial_result[0][lane];
                        if (col_index < output_size) {
                            float2 bias_f = __half22float2(bias_cast[col_index]);
                            float2 sum_g = __half22float2(sum[0]);
                            sum_g.x += bias_f.x;
                            sum_g.y += bias_f.y;
                            output_cast[(j + t) * output_size + col_index] =
                                __float22half2_rn(sum_g);
                        }
                    }
                }
            }
        }
        weight_cast = reinterpret_cast<const int16_t*>(weight);
    }
#endif
}

__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const __half* weight,
                                        const __half* bias,
                                        const __half* gamma,
                                        const __half* beta,
                                        const float epsilon,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ __half2 input_shared[9000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                __half2 inp_reg[8];
                int k = 0;
                int input_id = id;  //(gid + warp_num * lane);
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                // b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    sum += inp_f.x + inp_f.y;
                }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    inp_f.x -= mean;
                    inp_f.y -= mean;
                    inp_reg[f] = __float22half2_rn(inp_f);
                    sum += inp_f.x * inp_f.x;
                    sum += inp_f.y * inp_f.y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                __half2 variance_h = __float2half2_rn(sum);
                const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
                const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f] = inp_reg[f] * variance_h;
                    inp_reg[f] = inp_reg[f] * gamma_cast[id] + beta_cast[id];
                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        {
            int wid = gid << 2;
            int offset = wid * output_size;
            float2 sum[INPUT_TILE];
            for (int t = 0; t < INPUT_TILE; t++) {
                sum[t].x = 0;
                sum[t].y = 0;
            }

            while (wid < hidden_dim) {
                __half2 vals_f[4 * INPUT_TILE];
                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        __half2 val_h[2];
                        val_h[0] = input_shared[t * hidden_half + (wid >> 1)];
                        val_h[1] = input_shared[t * hidden_half + (wid >> 1) + 1];

                        __half* inp_data[2];
                        inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                        inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                        vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                        vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                        vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                        vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                    }
                }

                int row = blockIdx.x * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    __half2 weight_h[4];
                    weight_h[0] = weight_cast[offset1];
                    weight_h[1] = weight_cast[output_size + offset1];
                    weight_h[2] = weight_cast[(output_size << 1) + offset1];
                    weight_h[3] = weight_cast[((output_size << 1) + output_size) + offset1];

#pragma unroll
                    for (int t = 0; t < INPUT_TILE; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[4];
                            mul[0] = __half22float2(vals_f[(t << 2)] * weight_h[0]);
                            mul[1] = __half22float2(vals_f[(t << 2) + 1] * weight_h[1]);
                            mul[2] = __half22float2(vals_f[(t << 2) + 2] * weight_h[2]);
                            mul[3] = __half22float2(vals_f[(t << 2) + 3] * weight_h[3]);

                            sum[t].x += mul[0].x + mul[1].x + mul[2].x + mul[3].x;
                            sum[t].y += mul[0].y + mul[1].y + mul[2].y + mul[3].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 4;
                offset += (output_size * warp_num * 4);
            }

            __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
            const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    //__syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[0][gid] = sum_g; }
                    __syncthreads();

                    if (gid == 0) {
                        sum_g = partial_result[0][lane];
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            float2 bias_f = __half22float2(bias_cast[col]);
                            sum_g.x += bias_f.x;
                            sum_g.y += bias_f.y;
                            output_cast[(j + t) * output_size + col] = __float22half2_rn(sum_g);
                        }
                    }
                }
            }
        }
    }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output,
                                        const float* vals,
                                        const float* weight,
                                        const float* bias,
                                        const float* gamma,
                                        const float* beta,
                                        const float epsilon,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ float2 input_shared[5000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 inp_reg[8];
                int k = 0;          // Check if k goes from 0 to 7
                int input_id = id;  //(gid + warp_num * lane);
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];

                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) { sum += inp_reg[f].x + inp_reg[f].y; }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    inp_reg[f].x -= mean;
                    inp_reg[f].y -= mean;
                    sum += inp_reg[f].x * inp_reg[f].x;
                    sum += inp_reg[f].y * inp_reg[f].y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                const float2* gamma_cast = reinterpret_cast<const float2*>(gamma);
                const float2* beta_cast = reinterpret_cast<const float2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f].x = inp_reg[f].x * sum;
                    inp_reg[f].y = inp_reg[f].y * sum;

                    inp_reg[f].x = inp_reg[f].x * gamma_cast[id].x + beta_cast[id].x;
                    inp_reg[f].y = inp_reg[f].y * gamma_cast[id].y + beta_cast[id].y;

                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        {
            int wid = gid << 1;
            int offset = wid * output_size;
            float2 sum[INPUT_TILE];
            for (int t = 0; t < INPUT_TILE; t++) {
                sum[t].x = 0;
                sum[t].y = 0;
            }

            while (wid < hidden_dim) {
                float2 val_data[INPUT_TILE];
                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        val_data[t] = input_shared[t * hidden_half + (wid >> 1)];
                    }
                }

                int row = blockIdx.x * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    float2 weight[2];
                    weight[0] = weight_cast[offset1];
                    weight[1] = weight_cast[output_size + offset1];

                    for (int t = 0; t < INPUT_TILE; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[2];
                            mul[0].x = val_data[t].x * weight[0].x;
                            mul[0].y = val_data[t].x * weight[0].y;
                            mul[1].x = val_data[t].y * weight[1].x;
                            mul[1].y = val_data[t].y * weight[1].y;

                            sum[t].x += mul[0].x + mul[1].x;
                            sum[t].y += mul[0].y + mul[1].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 2;
                offset += (output_size * warp_num * 2);
            }

            __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
            const float2* bias_cast = reinterpret_cast<const float2*>(bias);
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    //__syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[0][gid] = sum_g; }
                    __syncthreads();

                    if (gid == 0) {
                        sum_g = partial_result[0][lane];
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            float2 bias_f = bias_cast[col];
                            sum_g.x += bias_f.x;
                            sum_g.y += bias_f.y;
                            output_cast[(j + t) * output_size + col] = sum_g;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    const float epsilon,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, weight, bias, gamma, beta, epsilon, hidden_dim, input_size, output_size);
}

template void launch_input_tiled_gemm_kernel(float* output,
                                             const float* vals,
                                             const float* weight,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const __half* weight,
                                             const __half* bias,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const int8_t* weight,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    const float epsilon,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    float* scale,
                                    int groups,
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                vals,
                                                                weight,
                                                                bias,
                                                                gamma,
                                                                beta,
                                                                epsilon,
                                                                hidden_dim,
                                                                input_size,
                                                                output_size,
                                                                scale,
                                                                groups);
}

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const int8_t* weight,
                                             const __half* bias,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             float* scale,
                                             int groups,
                                             cudaStream_t stream);

__global__ void input_tiled_gemm_kernel_gelu(__half* output,
                                             __half* residual_add,
                                             const __half* vals,
                                             const __half* residual,
                                             const __half* input_bias,
                                             const int8_t* weight,
                                             const __half* bias,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             float* qscale,
                                             int groups)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);
    __half2* residual_add_cast = reinterpret_cast<__half2*>(residual_add);
    const int16_t* weight_cast = reinterpret_cast<const int16_t*>(weight);
    const __half2* input_bias_cast = reinterpret_cast<const __half2*>(input_bias);

    int hidden_half = hidden_dim >> 1;

    int quantization_stride = (hidden_dim * (output_size << 1)) / groups;
    __shared__ __half shared_quantize_scale[MAX_QUANTIZE_GROUPING];
    // reading all the quantization scale into a small shared buffer
    if (threadIdx.x < groups)
        shared_quantize_scale[threadIdx.x] = __float2half(qscale[threadIdx.x]);
    __syncthreads();
    int col_index = blockIdx.x * WARP_SIZE + lane;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ __half2 input_shared[9000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                __half2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    float2 inp_f = __half22float2(inp_reg[k]);
                    float2 residual_f =
                        __half22float2(residual_cast[(j + t) * hidden_half + input_id]);
                    float2 bias_f = __half22float2(input_bias_cast[input_id]);
                    inp_f.x += residual_f.x + bias_f.x;
                    inp_f.y += residual_f.y + bias_f.y;
                    inp_reg[k] = __float22half2_rn(inp_f);
                    residual_add_cast[(j + t) * hidden_half + input_id] = inp_reg[k];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    sum += inp_f.x + inp_f.y;
                }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    inp_f.x -= mean;
                    inp_f.y -= mean;
                    inp_reg[f] = __float22half2_rn(inp_f);
                    sum += inp_f.x * inp_f.x;
                    sum += inp_f.y * inp_f.y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                __half2 variance_h = __float2half2_rn(sum);
                const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
                const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f] = inp_reg[f] * h2rsqrt(variance_h);
                    inp_reg[f] = inp_reg[f] * gamma_cast[id] + beta_cast[id];
                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        int wid = gid << 2;
        weight_cast += (wid * output_size + col_index);

        __half2 sum[INPUT_TILE];
        for (int t = 0; t < INPUT_TILE; t++) { sum[t] = __float2half2_rn(0.f); }

        while (wid < hidden_dim) {
            // updating the quantization scale

            __half2 qscale_data;
            {
                auto tmp = shared_quantize_scale[0];
                qscale_data = __halves2half2(tmp, tmp);
                if (groups > 1) {
                    unsigned index;
                    index = wid + (col_index << 1) * hidden_dim;
                    qscale_data = __halves2half2(
                        shared_quantize_scale[((index / quantization_stride))],
                        shared_quantize_scale[((index + hidden_dim) / quantization_stride)]);
                }
            }
            __half2 vals_f[INPUT_TILE * 4];
            for (int t = 0; t < INPUT_TILE; t++) {
                __half2 val_h[2];
                val_h[0] = input_shared[t * hidden_half + (wid >> 1)];
                val_h[1] = input_shared[t * hidden_half + (wid >> 1) + 1];

                __half* inp_data[2];
                inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
            }

            if (col_index < output_size) {
                int16_t weight_q[loop_unroll];
#pragma unroll
                for (int k = 0; k < loop_unroll; k++) weight_q[k] = weight_cast[k * output_size];

#pragma unroll
                for (int t = 0; t < INPUT_TILE; t++) {
#pragma unroll
                    for (int li = 0; li < loop_unroll; li++) {
                        int8_t* weight_8 = reinterpret_cast<int8_t*>(&weight_q[li]);
                        float2 weight_f;
                        weight_f.x = (float)weight_8[0];
                        weight_f.y = (float)weight_8[1];
                        auto mul =
                            __float22half2_rn(weight_f) * qscale_data * vals_f[(t << 2) + li];
                        if (ACC_HALF)
                            sum[t] += mul;
                        else {
                            float2 mul_f = __half22float2(mul);
                            float2 sum_f = __half22float2(sum[t]);
                            sum_f.x += mul_f.x;
                            sum_f.y += mul_f.y;
                            sum[t] = __float22half2_rn(sum_f);
                        }
                    }
                }
            }
            wid += (warp_num << loop_unroll_bits);
            weight_cast += ((warp_num << loop_unroll_bits) * output_size);
        }
        {
            __shared__ __half2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
            const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);

            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    partial_result[gid][lane] = sum[t];
                    __syncthreads();
                    sum[t] = partial_result[lane][gid];

                    if (ACC_HALF) {
#pragma unroll
                        for (int i = 1; i < WARP_SIZE; i *= 2) {
                            float* sum_f = reinterpret_cast<float*>(&sum[t]);
                            float temp = g.shfl_xor(*sum_f, i);
                            __half2* sum_h = reinterpret_cast<__half2*>(&temp);
                            sum[t] += *sum_h;
                        }
                    } else {
                        float2 sum_g = __half22float2(sum[t]);
#pragma unroll
                        for (int i = 1; i < WARP_SIZE; i *= 2) {
                            sum_g.x += g.shfl_xor(sum_g.x, i);
                            sum_g.y += g.shfl_xor(sum_g.y, i);
                        }
                        sum[t] = __float22half2_rn(sum_g);
                    }

                    if (lane == 0) { partial_result[0][gid] = sum[t]; }
                    __syncthreads();

                    if (gid == 0) {
                        if (col_index < output_size) {
                            float2 sum_g = __half22float2(partial_result[0][lane]);
                            float2 bias_f = __half22float2(bias_cast[col_index]);
                            sum_g.x = bias_f.x + sum_g.x;
                            sum_g.y = bias_f.y + sum_g.y;
                            sum_g.x = gelu(sum_g.x);
                            sum_g.y = gelu(sum_g.y);

                            output_cast[(j + t) * output_size + col_index] =
                                __float22half2_rn(sum_g);
                        }
                    }
                }
            }
        }
        weight_cast = reinterpret_cast<const int16_t*>(weight);
    }
#endif
}

__global__ void input_tiled_gemm_kernel_gelu(__half* output,
                                             __half* residual_add,
                                             const __half* vals,
                                             const __half* residual,
                                             const __half* input_bias,
                                             const __half* weight,
                                             const __half* bias,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);
    __half2* residual_add_cast = reinterpret_cast<__half2*>(residual_add);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);
    const __half2* input_bias_cast = reinterpret_cast<const __half2*>(input_bias);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ __half2 input_shared[9000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                __half2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    float2 inp_f = __half22float2(inp_reg[k]);
                    float2 residual_f =
                        __half22float2(residual_cast[(j + t) * hidden_half + input_id]);
                    float2 bias_f = __half22float2(input_bias_cast[input_id]);
                    inp_f.x += residual_f.x + bias_f.x;
                    inp_f.y += residual_f.y + bias_f.y;
                    inp_reg[k] = __float22half2_rn(inp_f);
                    residual_add_cast[(j + t) * hidden_half + input_id] = inp_reg[k];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                // b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    sum += inp_f.x + inp_f.y;
                }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    inp_f.x -= mean;
                    inp_f.y -= mean;
                    inp_reg[f] = __float22half2_rn(inp_f);
                    sum += inp_f.x * inp_f.x;
                    sum += inp_f.y * inp_f.y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                __half2 variance_h = __float2half2_rn(sum);
                const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
                const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
                for (int f = 0; f < k; f++) {
                    int tid = f * blockDim.x + threadIdx.x;
                    inp_reg[f] = inp_reg[f] * variance_h;
                    inp_reg[f] = inp_reg[f] * gamma_cast[tid] + beta_cast[tid];
                    input_shared[tid + t * hidden_half] = inp_reg[f];
                    // output_cast[(j + t) * hidden_half + tid] = inp_reg[f];
                }
                b.sync();
            }
        }

        int wid = gid << 2;
        int offset = wid * output_size;
        float2 sum[INPUT_TILE];
        for (int t = 0; t < INPUT_TILE; t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        while (wid < hidden_dim) {
            __half2 vals_f[INPUT_TILE * 4];
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    __half2 val_h[2];
                    val_h[0] = input_shared[t * hidden_half + (wid >> 1)];
                    val_h[1] = input_shared[t * hidden_half + (wid >> 1) + 1];

                    __half* inp_data[2];
                    inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                    inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                    vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                    vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                    vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                    vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                }
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                __half2 weight_h[4];
                weight_h[0] = weight_cast[offset1];
                weight_h[1] = weight_cast[output_size + offset1];
                weight_h[2] = weight_cast[(output_size << 1) + offset1];
                weight_h[3] = weight_cast[((output_size << 1) + output_size) + offset1];
#pragma unroll
                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        float2 mul[4];
                        mul[0] = __half22float2(vals_f[(t << 2)] * weight_h[0]);
                        mul[1] = __half22float2(vals_f[(t << 2) + 1] * weight_h[1]);
                        mul[2] = __half22float2(vals_f[(t << 2) + 2] * weight_h[2]);
                        mul[3] = __half22float2(vals_f[(t << 2) + 3] * weight_h[3]);

                        sum[t].x += mul[0].x + mul[1].x + mul[2].x + mul[3].x;
                        sum[t].y += mul[0].y + mul[1].y + mul[2].y + mul[3].y;
                    }
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 4;
            offset += (output_size * warp_num * 4);
        }
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 sum_g = sum[t];
                __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
                const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);
                {
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    //__syncthreads();
                }

#pragma unroll
                for (int i = 1; i < WARP_SIZE; i *= 2) {
                    sum_g.x += g.shfl_xor(sum_g.x, i);
                    sum_g.y += g.shfl_xor(sum_g.y, i);
                }

                if (lane == 0) { partial_result[0][gid] = sum_g; }
                __syncthreads();

                if (gid == 0) {
                    int col = blockIdx.x * WARP_SIZE + lane;
                    if (col < output_size) {
                        sum_g = partial_result[0][lane];
                        float2 bias_f = __half22float2(bias_cast[col]);
                        sum_g.x = bias_f.x + sum_g.x;
                        sum_g.y = bias_f.y + sum_g.y;
                        sum_g.x = gelu(sum_g.x);
                        sum_g.y = gelu(sum_g.y);

                        output_cast[(j + t) * output_size + col] = __float22half2_rn(sum_g);
                    }
                }
            }
        }
    }
#endif
}

__global__ void input_tiled_gemm_kernel_gelu(float* output,
                                             float* residual_add,
                                             const float* vals,
                                             const float* residual,
                                             const float* input_bias,
                                             const float* weight,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);
    float2* residual_add_cast = reinterpret_cast<float2*>(residual_add);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);
    const float2* input_bias_cast = reinterpret_cast<const float2*>(input_bias);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ float2 input_shared[5000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    float2 residual_f = residual_cast[(j + t) * hidden_half + input_id];
                    float2 bias_f = input_bias_cast[input_id];
                    inp_reg[k].x += residual_f.x + bias_f.x;
                    inp_reg[k].y += residual_f.y + bias_f.y;
                    residual_add_cast[(j + t) * hidden_half + input_id] = inp_reg[k];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) { sum += inp_reg[f].x + inp_reg[f].y; }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    inp_reg[f].x -= mean;
                    inp_reg[f].y -= mean;
                    sum += inp_reg[f].x * inp_reg[f].x;
                    sum += inp_reg[f].y * inp_reg[f].y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                const float2* gamma_cast = reinterpret_cast<const float2*>(gamma);
                const float2* beta_cast = reinterpret_cast<const float2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f].x = inp_reg[f].x * sum;
                    inp_reg[f].y = inp_reg[f].y * sum;

                    inp_reg[f].x = inp_reg[f].x * gamma_cast[id].x + beta_cast[id].x;
                    inp_reg[f].y = inp_reg[f].y * gamma_cast[id].y + beta_cast[id].y;

                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        int wid = gid << 1;
        int offset = wid * output_size;
        float2 sum[INPUT_TILE];
        for (int t = 0; t < INPUT_TILE; t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        while (wid < hidden_dim) {
            float2 val_data[INPUT_TILE];
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    val_data[t] = input_shared[t * hidden_half + (wid >> 1)];
                }
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                float2 weight[2];
                weight[0] = weight_cast[offset1];
                weight[1] = weight_cast[output_size + offset1];

                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        float2 mul[2];
                        mul[0].x = val_data[t].x * weight[0].x;
                        mul[0].y = val_data[t].x * weight[0].y;
                        mul[1].x = val_data[t].y * weight[1].x;
                        mul[1].y = val_data[t].y * weight[1].y;

                        sum[t].x += mul[0].x + mul[1].x;
                        sum[t].y += mul[0].y + mul[1].y;
                    }
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 2;
            offset += (output_size * warp_num * 2);
        }
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 sum_g = sum[t];
                __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
                const float2* bias_cast = reinterpret_cast<const float2*>(bias);
                {
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    __syncthreads();
                }

#pragma unroll
                for (int i = 1; i < WARP_SIZE; i *= 2) {
                    sum_g.x += g.shfl_xor(sum_g.x, i);
                    sum_g.y += g.shfl_xor(sum_g.y, i);
                }

                if (lane == 0) { partial_result[0][gid] = sum_g; }
                __syncthreads();

                if (gid == 0) {
                    int col = blockIdx.x * WARP_SIZE + lane;
                    if (col < output_size) {
                        sum_g = partial_result[0][lane];
                        float2 bias_f = bias_cast[col];
                        sum_g.x = bias_f.x + sum_g.x;
                        sum_g.y = bias_f.y + sum_g.y;
                        sum_g.x = gelu(sum_g.x);
                        sum_g.y = gelu(sum_g.y);

                        output_cast[(j + t) * output_size + col] = sum_g;
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel_gelu(T* output,
                                         T* residual_add,
                                         const T* vals,
                                         const T* residual,
                                         const T* input_bias,
                                         const T* weight,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         const float epsilon,
                                         int hidden_dim,
                                         int input_size,
                                         int output_size,
                                         cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel_gelu<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                     residual_add,
                                                                     vals,
                                                                     residual,
                                                                     input_bias,
                                                                     weight,
                                                                     bias,
                                                                     gamma,
                                                                     beta,
                                                                     epsilon,
                                                                     hidden_dim,
                                                                     input_size,
                                                                     output_size);
}

template void launch_input_tiled_gemm_kernel_gelu(float* output,
                                                  float* residual_add,
                                                  const float* vals,
                                                  const float* residual,
                                                  const float* input_bias,
                                                  const float* weight,
                                                  const float* bias,
                                                  const float* gamma,
                                                  const float* beta,
                                                  const float epsilon,
                                                  int hidden_dim,
                                                  int input_size,
                                                  int output_size,
                                                  cudaStream_t stream);

template void launch_input_tiled_gemm_kernel_gelu(__half* output,
                                                  __half* residual_add,
                                                  const __half* vals,
                                                  const __half* residual,
                                                  const __half* input_bias,
                                                  const __half* weight,
                                                  const __half* bias,
                                                  const __half* gamma,
                                                  const __half* beta,
                                                  const float epsilon,
                                                  int hidden_dim,
                                                  int input_size,
                                                  int output_size,
                                                  cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel_gelu(T* output,
                                         T* residual_add,
                                         const T* vals,
                                         const T* residual,
                                         const T* input_bias,
                                         const int8_t* weight,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         const float epsilon,
                                         int hidden_dim,
                                         int input_size,
                                         int output_size,
                                         float* scale,
                                         int groups,
                                         cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel_gelu<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                     residual_add,
                                                                     vals,
                                                                     residual,
                                                                     input_bias,
                                                                     weight,
                                                                     bias,
                                                                     gamma,
                                                                     beta,
                                                                     epsilon,
                                                                     hidden_dim,
                                                                     input_size,
                                                                     output_size,
                                                                     scale,
                                                                     groups);
}

template void launch_input_tiled_gemm_kernel_gelu(__half* output,
                                                  __half* residual_add,
                                                  const __half* vals,
                                                  const __half* residual,
                                                  const __half* input_bias,
                                                  const int8_t* weight,
                                                  const __half* bias,
                                                  const __half* gamma,
                                                  const __half* beta,
                                                  const float epsilon,
                                                  int hidden_dim,
                                                  int input_size,
                                                  int output_size,
                                                  float* scale,
                                                  int groups,
                                                  cudaStream_t stream);
