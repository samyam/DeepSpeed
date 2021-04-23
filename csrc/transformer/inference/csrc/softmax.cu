
#include <limits>
#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define Attn_Threads 128
#define Reduce_Threads 32
#define attn_warps 4
#define MAX_ATTN_REG 4  // MAX Head Size 256

#define SCALE_HALF true

#define minus_infinity -1 * std::numeric_limits<float>::infinity()

void CheckCudaErrorAux(const char* file, unsigned line)
{
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) return;
    std::cerr << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line
              << std::endl;
    throw std::runtime_error("CUDA ERROR!!!\n");
}

#define CUDA_CHECK_ERROR() CheckCudaErrorAux(__FILE__, __LINE__)

namespace cg = cooperative_groups;

template <int tbSize, int tbSeq>
__global__ void attn_softmax_v2(__half* vals,
                                __half* mask,
                                bool triangular,
                                int total_count,
                                int heads,
                                int sequence_length,
                                int num_seq,
                                float scale)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    float2 low_data[tbSeq];
    float2 high_data[tbSeq];

    __half2 h_scale = __float2half2_rn(scale);

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + wid;

    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        int mask_offset = (iter_offset / (heads * num_seq)) * (sequence_length);
        int seq_id = iter_offset % num_seq;
        int seq_id4 = seq_id >> 2;
        float max_val = minus_infinity;

        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);
            if ((!triangular || ((data_id >> 2) <= seq_id4)) && data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    low_data[i].x = __half2float(vals[data_id]);
                    low_data[i].y = (!triangular || ((data_id + 1) <= seq_id))
                                        ? __half2float(vals[data_id + 1])
                                        : minus_infinity;
                    high_data[i].x = (!triangular || ((data_id + 2) <= seq_id))
                                         ? __half2float(vals[data_id + 2])
                                         : minus_infinity;
                    high_data[i].y = (!triangular || ((data_id + 3) <= seq_id))
                                         ? __half2float(vals[data_id + 3])
                                         : minus_infinity;
                } else {
                    low_data[i].x = __half2float(vals[data_id]);
                    low_data[i].y = (((!triangular || (data_id + 1) <= seq_id)) &&
                                     (data_id + 1) < sequence_length)
                                        ? __half2float(vals[data_id + 1])
                                        : minus_infinity;
                    high_data[i].x = (((!triangular || (data_id + 2) <= seq_id)) &&
                                      (data_id + 2) < sequence_length)
                                         ? __half2float(vals[data_id + 2])
                                         : minus_infinity;
                    high_data[i].y = minus_infinity;
                }
                max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
                max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
            } else {
                low_data[i].x = minus_infinity;
                low_data[i].y = minus_infinity;
                high_data[i].x = minus_infinity;
                high_data[i].y = minus_infinity;
            }
        }

        for (int i = 1; i < tbSize; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        float sum = 0;
        for (int i = 0; i < tbSeq; i++) {
            low_data[i].x = __expf(low_data[i].x - max_val);
            low_data[i].y = __expf(low_data[i].y - max_val);
            high_data[i].x = __expf(high_data[i].x - max_val);
            high_data[i].y = __expf(high_data[i].y - max_val);

            sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        }

        for (int i = 1; i < tbSize; i *= 2) sum += g.shfl_xor(sum, i);

        sum += 1e-6;

        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    vals[data_id] = low_data[i].x / sum;
                    vals[data_id + 1] = low_data[i].y / sum;
                    vals[data_id + 2] = high_data[i].x / sum;
                    vals[data_id + 3] = high_data[i].y / sum;
                } else {
                    vals[data_id] = low_data[i].x / sum;
                    if ((data_id + 1) < sequence_length) vals[data_id + 1] = low_data[i].y / sum;
                    if ((data_id + 2) < sequence_length) vals[data_id + 2] = high_data[i].x / sum;
                }
            }
        }
    }
#endif
}

template <int tbSize, int tbSeq>
__global__ void attn_softmax_v2(float* vals,
                                float* attn_mask,
                                bool triangular,
                                int total_count,
                                int heads,
                                int sequence_length,
                                int num_seq,
                                float scale)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    float4 data[tbSeq];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int iter_offset = blockIdx.x * warp_num + wid;
    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        // int mask_offset = (iter_offset / (heads * num_seq)) * (sequence_length);
        int seq_id = iter_offset % num_seq;
        int seq_id4 = seq_id >> 2;
        float max_val = minus_infinity;

        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);
            if ((!triangular || ((data_id >> 2) <= seq_id4)) && data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    data[i].x = (vals[data_id]);
                    data[i].y = (!triangular || ((data_id + 1) <= seq_id)) ? (vals[data_id + 1])
                                                                           : minus_infinity;
                    data[i].z = (!triangular || ((data_id + 2) <= seq_id)) ? (vals[data_id + 2])
                                                                           : minus_infinity;
                    data[i].w = (!triangular || ((data_id + 3) <= seq_id)) ? (vals[data_id + 3])
                                                                           : minus_infinity;
                } else {
                    data[i].x = (vals[data_id]);
                    data[i].y = (((!triangular || (data_id + 1) <= seq_id)) &&
                                 (data_id + 1) < sequence_length)
                                    ? (vals[data_id + 1])
                                    : minus_infinity;
                    data[i].z = (((!triangular || (data_id + 2) <= seq_id)) &&
                                 (data_id + 2) < sequence_length)
                                    ? (vals[data_id + 2])
                                    : minus_infinity;
                    data[i].w = minus_infinity;
                }
                max_val = (data[i].x > max_val ? data[i].x : max_val);
                max_val = (data[i].y > max_val ? data[i].y : max_val);
                max_val = (data[i].z > max_val ? data[i].z : max_val);
                max_val = (data[i].w > max_val ? data[i].w : max_val);
            } else {
                data[i].x = minus_infinity;
                data[i].y = minus_infinity;
                data[i].z = minus_infinity;
                data[i].w = minus_infinity;
            }
        }

        for (int i = 1; i < tbSize; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        float sum = 0;
        for (int i = 0; i < tbSeq; i++) {
            data[i].x = __expf(data[i].x - max_val);
            data[i].y = __expf(data[i].y - max_val);
            data[i].z = __expf(data[i].z - max_val);
            data[i].w = __expf(data[i].w - max_val);

            sum += (data[i].x + data[i].y + data[i].z + data[i].w);
        }

        for (int i = 1; i < tbSize; i *= 2) sum += g.shfl_xor(sum, i);

        sum += 1e-6;

        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    vals[data_id] = data[i].x / sum;
                    vals[data_id + 1] = data[i].y / sum;
                    vals[data_id + 2] = data[i].z / sum;
                    vals[data_id + 3] = data[i].w / sum;
                } else {
                    vals[data_id] = data[i].x / sum;
                    if ((data_id + 1) < sequence_length) vals[data_id + 1] = data[i].y / sum;
                    if ((data_id + 2) < sequence_length) vals[data_id + 2] = data[i].z / sum;
                }
            }
        }
    }
}

template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            bool triangular,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            float scale,
                            cudaStream_t stream)
{
    int total_count = batch_size * heads * num_seq;
    dim3 grid_dim((total_count - 1) / 32 + 1);
    dim3 block_dim(1024);
    if (sequence_length <= 128)
        attn_softmax_v2<32, 1><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else if (sequence_length <= 256)
        attn_softmax_v2<32, 2><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else if (sequence_length <= 512)
        attn_softmax_v2<32, 4><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else if (sequence_length <= 1024)
        attn_softmax_v2<32, 8><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else if (sequence_length <= 2048)
        attn_softmax_v2<32, 16><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else if (sequence_length <= 4096)
        attn_softmax_v2<32, 32><<<grid_dim, block_dim, 0, stream>>>(
            vals, mask, triangular, total_count, heads, sequence_length, num_seq, scale);
    else
        throw std::runtime_error(
            "Unsupport Seq_Length! Check the restriction of the max_threads and "
            "max_thread_iterations!");

    CUDA_CHECK_ERROR();
}

template void launch_attn_softmax_v2(float* vals,
                                     float* mask,
                                     bool triangular,
                                     int batch_size,
                                     int heads,
                                     int num_seq,
                                     int sequence_length,
                                     float scale,
                                     cudaStream_t stream);
template void launch_attn_softmax_v2(__half* vals,
                                     __half* mask,
                                     bool triangular,
                                     int batch_size,
                                     int heads,
                                     int num_seq,
                                     int sequence_length,
                                     float scale,
                                     cudaStream_t stream);

__device__ void attn_score(__half* shared_soft,
                           __half* query,
                           __half* key,
                           __half* new_key,
                           __half* key_merged,
                           bool merging,
                           float norm_factor,
                           int head_size,
                           int total_count,
                           int num_seq,
                           int hidden,
                           int value_length)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2 queries_low[MAX_ATTN_REG];
    __half2 queries_high[MAX_ATTN_REG];
    float2* query_cast = reinterpret_cast<float2*>(query);
    float2* key_cast = reinterpret_cast<float2*>(key);
    float2* key_merged_cast;
    if (merging) key_merged_cast = reinterpret_cast<float2*>(key_merged);
    float2* new_key_cast;
    if (new_key) new_key_cast = reinterpret_cast<float2*>(new_key);
    __half2 norm_factor_h = __float2half2_rn(norm_factor);
    int inp_size = head_size >> 1;
    int input_offset = (blockIdx.x * warp_num + wid);

    if (input_offset < total_count) {
        input_offset =
            (input_offset % num_seq) * (hidden >> 1) + (input_offset / num_seq) * inp_size;

        int row = lane;
        int p = 0;

        while (row < inp_size) {
            float2 querie = query_cast[input_offset + row];

            __half2* query_value = reinterpret_cast<__half2*>(&querie);
            if (SCALE_HALF) {
                queries_low[p] = query_value[0] / norm_factor_h;
                queries_high[p] = query_value[1] / norm_factor_h;
            } else {
                float2 query_value_f[2];
                query_value_f[0] = __half22float2(query_value[0]);
                query_value_f[1] = __half22float2(query_value[1]);
                query_value_f[0].x /= norm_factor;
                query_value_f[1].x /= norm_factor;
                query_value_f[0].y /= norm_factor;
                query_value_f[1].y /= norm_factor;
                queries_low[p] = __float22half2_rn(query_value_f[0]);
                queries_high[p] = __float22half2_rn(query_value_f[1]);
            }
            p++;
            row += WARP_SIZE;
        }

        int seq_key = (blockIdx.x * warp_num + wid) / num_seq;

        int unique_id = (blockIdx.x * warp_num + wid) % num_seq;

        key_cast += (seq_key * inp_size);
        key_merged_cast += (seq_key * inp_size);

        int key_size = total_count / num_seq;
        int score_index = 0;

        if (seq_key < key_size) {
            {
                float scores[WARP_SIZE];
                int warp_iter;
                for (int i = 0; i < value_length; i += WARP_SIZE) {
                    warp_iter = (value_length - i) > WARP_SIZE ? WARP_SIZE : (value_length - i);
#pragma unroll
                    for (int p = 0; p < warp_iter; p++) { scores[p] = 0; }

                    for (int k = 0; k < warp_iter; k++) {
                        row = lane;
                        int p = 0;
                        while (row < inp_size) {
                            float2 key_value_reg = key_cast[row];
                            if (merging && unique_id == 0) key_merged_cast[row] = key_value_reg;
                            __half2* key_value = reinterpret_cast<__half2*>(&key_value_reg);
                            if (SCALE_HALF) {
                                key_value[0] /= norm_factor_h;
                                key_value[1] /= norm_factor_h;
                            } else {
                                float2 key_value_f[2];
                                key_value_f[0] = __half22float2(key_value[0]);
                                key_value_f[1] = __half22float2(key_value[1]);

                                key_value_f[0].x /= norm_factor;
                                key_value_f[1].x /= norm_factor;
                                key_value_f[0].y /= norm_factor;
                                key_value_f[1].y /= norm_factor;
                                key_value[0] = __float22half2_rn(key_value_f[0]);
                                key_value[1] = __float22half2_rn(key_value_f[1]);
                            }

                            float2 mul[2];
                            mul[0] = __half22float2(queries_low[p] * key_value[0]);
                            mul[1] = __half22float2(queries_high[p] * key_value[1]);
                            scores[k] += (mul[0].x + mul[0].y) + (mul[1].x + mul[1].y);
                            row += WARP_SIZE;
                            p++;
                        }
                        key_cast += (hidden >> 1);
                        if (merging && unique_id == 0) key_merged_cast += (hidden >> 1);
#pragma unroll
                        for (int w = 1; w < WARP_SIZE; w *= 2)
                            scores[k] += g.shfl_xor(scores[k], w);
                    }
                    if (lane < (warp_iter >> 1)) {
                        shared_soft[wid * 1024 + ((lane + score_index) << 1)] =
                            __float2half(scores[(lane << 1)]);
                        shared_soft[wid * 1024 + ((lane + score_index) << 1) + 1] =
                            __float2half(scores[(lane << 1) + 1]);
                    }
                    score_index += (warp_iter >> 1);
                }
                if (warp_iter % 2 == 1) {
                    if (lane == (warp_iter >> 1)) {
                        shared_soft[wid * 1024 + (score_index << 1)] =
                            __float2half(scores[(lane << 1)]);
                    }
                }
            }
            if (new_key) {
                new_key_cast += (((blockIdx.x * warp_num + wid) / num_seq) * inp_size);
                if (merging) {
                    key_merged_cast = reinterpret_cast<float2*>(key_merged);
                    key_merged_cast += ((blockIdx.x * warp_num + wid) / num_seq) * inp_size +
                                       ((hidden >> 1) * value_length);
                }
                row = lane;
                int p = 0;
                float score = 0;
                while (row < inp_size) {
                    float2 new_key_data = new_key_cast[row];
                    if (merging && unique_id == 0) key_merged_cast[row] = new_key_data;
                    __half2* key_value = reinterpret_cast<__half2*>(&new_key_data);
                    if (SCALE_HALF) {
                        key_value[0] /= norm_factor_h;
                        key_value[1] /= norm_factor_h;
                    } else {
                        float2 key_value_f[2];
                        key_value_f[0] = __half22float2(key_value[0]);
                        key_value_f[1] = __half22float2(key_value[1]);

                        key_value_f[0].x /= norm_factor;
                        key_value_f[1].x /= norm_factor;
                        key_value_f[0].y /= norm_factor;
                        key_value_f[1].y /= norm_factor;
                        key_value[0] = __float22half2_rn(key_value_f[0]);
                        key_value[1] = __float22half2_rn(key_value_f[1]);
                    }

                    float2 mul[2];
                    mul[0] = __half22float2(queries_low[p] * key_value[0]);
                    mul[1] = __half22float2(queries_high[p] * key_value[1]);
                    score += (mul[0].x + mul[0].y) + (mul[1].x + mul[1].y);
                    row += WARP_SIZE;
                    p++;
                }
#pragma unroll
                for (int w = 1; w < WARP_SIZE; w *= 2) score += g.shfl_down(score, w);

                if (lane == 0) {
                    if ((value_length + 1) % 2 == 0) {
                        shared_soft[wid * 1024 + (score_index << 1) + 1] = __float2half(score);
                    } else {
                        shared_soft[wid * 1024 + (score_index << 1)] = __float2half(score);
                    }
                }
            }
        }
    }
}

__device__ void attn_score(float* shared_soft,
                           float* query,
                           float* key,
                           float* new_key,
                           float* key_merged,
                           bool merging,
                           float norm_factor,
                           int head_size,
                           int total_count,
                           int num_seq,
                           int hidden,
                           int value_length)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2 query_value[8];
    float2* query_cast = reinterpret_cast<float2*>(query);
    float2* key_cast = reinterpret_cast<float2*>(key);
    float2* key_merged_cast;
    if (merging) key_merged_cast = reinterpret_cast<float2*>(key_merged);
    float2* new_key_cast;
    if (new_key) new_key_cast = reinterpret_cast<float2*>(new_key);

    int inp_size = head_size;
    int input_offset = (blockIdx.x * warp_num + wid);

    if (input_offset < total_count) {
        query_cast += (input_offset % num_seq) * (hidden) + (input_offset / num_seq) * inp_size;

        int row = lane;
        int p = 0;

        while (row < inp_size) {
            query_value[p] = query_cast[row];

            query_value[p].x /= norm_factor;
            query_value[p].y /= norm_factor;

            p++;
            row += WARP_SIZE;
        }

        int seq_key = input_offset / num_seq;

        int unique_id = input_offset % num_seq;

        key_cast += (seq_key * inp_size);
        key_merged_cast += (seq_key * inp_size);

        int key_size = total_count / num_seq;
        int score_index = 0;

        if (seq_key < key_size) {
            {
                float scores[WARP_SIZE];
                int warp_iter;
                for (int i = 0; i < value_length; i += WARP_SIZE) {
                    warp_iter = (value_length - i) > WARP_SIZE ? WARP_SIZE : (value_length - i);
#pragma unroll
                    for (int p = 0; p < warp_iter; p++) { scores[p] = 0; }

                    for (int k = 0; k < warp_iter; k++) {
                        row = lane;
                        int p = 0;
                        while (row < inp_size) {
                            float2 key_value = key_cast[row];
                            if (merging && unique_id == 0) key_merged_cast[row] = key_value;

                            key_value.x /= norm_factor;
                            key_value.y /= norm_factor;

                            float2 mul;
                            mul.x = query_value[p].x * key_value.x;
                            mul.y = query_value[p].y * key_value.y;
                            scores[k] += (mul.x + mul.y);
                            row += WARP_SIZE;
                            p++;
                        }
                        key_cast += (hidden);
                        if (merging && unique_id == 0) key_merged_cast += (hidden);
#pragma unroll
                        for (int w = 1; w < WARP_SIZE; w *= 2)
                            scores[k] += g.shfl_xor(scores[k], w);
                    }
                    if (lane < (warp_iter >> 1)) {
                        shared_soft[wid * 1000 + ((lane + score_index) << 1)] = scores[(lane << 1)];
                        shared_soft[wid * 1000 + ((lane + score_index) << 1) + 1] =
                            scores[(lane << 1) + 1];
                    }
                    score_index += (warp_iter >> 1);
                }
                if (warp_iter % 2 == 1) {
                    if (lane == (warp_iter >> 1)) {
                        shared_soft[wid * 1000 + (score_index << 1)] = scores[(lane << 1)];
                    }
                }
            }
            if (new_key) {
                new_key_cast += (((blockIdx.x * warp_num + wid) / num_seq) * inp_size);
                if (merging) {
                    key_merged_cast = reinterpret_cast<float2*>(key_merged);
                    key_merged_cast += ((blockIdx.x * warp_num + wid) / num_seq) * inp_size +
                                       ((hidden)*value_length);
                }
                row = lane;
                int p = 0;
                float score = 0;
                while (row < inp_size) {
                    float2 key_value = new_key_cast[row];
                    if (merging && unique_id == 0) key_merged_cast[row] = key_value;

                    key_value.x /= norm_factor;
                    key_value.y /= norm_factor;

                    float2 mul;
                    mul.x = query_value[p].x * key_value.x;
                    mul.y = query_value[p].y * key_value.y;
                    score += (mul.x + mul.y);
                    row += WARP_SIZE;
                    p++;
                }
#pragma unroll
                for (int w = 1; w < WARP_SIZE; w *= 2) score += g.shfl_down(score, w);

                if (lane == 0) {
                    if ((value_length + 1) % 2 == 0) {
                        shared_soft[wid * 1000 + (score_index << 1) + 1] = score;
                    } else {
                        shared_soft[wid * 1000 + (score_index << 1)] = score;
                    }
                }
            }
        }
    }
}

template <int tbSeq>
__device__ void attn_softmax(__half* shared_soft,
                             __half2* shared_soft1,
                             __half* mask,
                             int heads,
                             int total_count,
                             int num_seq,
                             int sequence_length,
                             bool triangular,
                             bool recompute)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    // int tbSeq = (sequence_length-1) / (WARP_SIZE << 2) + 1;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2 low_data[tbSeq];
    float2 high_data[tbSeq];

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + wid;
    if (iter_offset < total_count) {
        int iteration_stride = (blockDim.x >> 5) * gridDim.x;

        int seq_id = iter_offset % num_seq;
        int seq_id4 = seq_id >> 2;
        float max_val = minus_infinity;

        int mask_offset = (iter_offset / (heads * num_seq)) * (sequence_length);
        /**********
            [1 0 0 0]      [0 -inf -inf -inf]
            [1 1 0 0]      [0  0   -inf -inf]
            [1 1 1 0]      [0  0   0    -inf]
            [1 1 1 1]      [0  0   0       0]
        **********/
        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);
            if ((!triangular || ((data_id >> 2) <= seq_id4)) && data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    low_data[i].x = __half2float(shared_soft[wid * 1024 + data_id]);
                    low_data[i].y = (!triangular || ((data_id + 1) <= seq_id))
                                        ? __half2float(shared_soft[wid * 1024 + data_id + 1])
                                        : minus_infinity;
                    high_data[i].x = (!triangular || ((data_id + 2) <= seq_id))
                                         ? __half2float(shared_soft[wid * 1024 + data_id + 2])
                                         : minus_infinity;
                    high_data[i].y = (!triangular || ((data_id + 3) <= seq_id))
                                         ? __half2float(shared_soft[wid * 1024 + data_id + 3])
                                         : minus_infinity;
                    if (mask && !triangular && recompute) {
                        low_data[i].x += __half2float(mask[data_id + mask_offset]);
                        low_data[i].y += __half2float(mask[data_id + mask_offset + 1]);
                        high_data[i].x += __half2float(mask[data_id + mask_offset + 2]);
                        high_data[i].y += __half2float(mask[data_id + mask_offset + 3]);
                    }
                } else {
                    low_data[i].x = __half2float(shared_soft[wid * 1024 + data_id]);
                    low_data[i].y = (((!triangular || (data_id + 1) <= seq_id)) &&
                                     (data_id + 1) < sequence_length)
                                        ? __half2float(shared_soft[wid * 1024 + data_id + 1])
                                        : minus_infinity;
                    high_data[i].x = (((!triangular || (data_id + 2) <= seq_id)) &&
                                      (data_id + 2) < sequence_length)
                                         ? __half2float(shared_soft[wid * 1024 + data_id + 2])
                                         : minus_infinity;
                    high_data[i].y = minus_infinity;
                    if (mask && !triangular && recompute) {
                        low_data[i].x += __half2float(mask[data_id + mask_offset]);
                        if ((data_id + 1) < sequence_length)
                            low_data[i].y += __half2float(mask[data_id + mask_offset + 1]);
                        if ((data_id + 2) < sequence_length)
                            high_data[i].x += __half2float(mask[data_id + mask_offset + 2]);
                        // high_data[i].y += __half2float(mask[data_id + mask_offset + 3]);
                    }
                }
                max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
                max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
            } else {
                low_data[i].x = minus_infinity;
                low_data[i].y = minus_infinity;
                high_data[i].x = minus_infinity;
                high_data[i].y = minus_infinity;
            }
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        __shared__ float partialSum[MAX_WARP_NUM];

        if (Reduce_Threads > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            b.sync();

            if (lane < warp_num) max_val = partialSum[lane];

            int iters = warp_num;
            if (Reduce_Threads < iteration_stride) iters /= (iteration_stride / Reduce_Threads);

            for (int i = 1; i < iters; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }

        float sum = 0;
        for (int i = 0; i < tbSeq; i++) {
            low_data[i].x = __expf(low_data[i].x - max_val);
            low_data[i].y = __expf(low_data[i].y - max_val);
            high_data[i].x = __expf(high_data[i].x - max_val);
            high_data[i].y = __expf(high_data[i].y - max_val);

            sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum += g.shfl_xor(sum, i);

        if (Reduce_Threads > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            b.sync();

            if (lane < warp_num) sum = partialSum[lane];

            int iters = warp_num;
            if (Reduce_Threads < iteration_stride) iters /= (iteration_stride / Reduce_Threads);

            for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

            sum = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }
        sum += 1e-6;
        for (int i = 0; i < tbSeq; i++) {
            int data_id = i * (WARP_SIZE << 2) + (lane << 2);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    low_data[i].x /= sum;
                    low_data[i].y /= sum;
                    high_data[i].x /= sum;
                    high_data[i].y /= sum;
                } else {
                    low_data[i].x /= sum;
                    low_data[i].y = (((data_id + 1) < sequence_length) ? low_data[i].y / sum : 0.f);
                    high_data[i].x =
                        (((data_id + 2) < sequence_length) ? high_data[i].x / sum : 0.f);
                    high_data[i].y = 0;
                }
                shared_soft1[wid * 1024 + (data_id >> 1)] = __float22half2_rn(low_data[i]);
                shared_soft1[wid * 1024 + (data_id >> 1) + 1] = __float22half2_rn(high_data[i]);
            }
        }
    }
}

template <int tbSeq>
__device__ void attn_softmax(float* shared_soft,
                             float2* shared_soft1,
                             float* mask,
                             int heads,
                             int total_count,
                             int num_seq,
                             int sequence_length,
                             bool triangular,
                             bool recompute)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    constexpr int reg_size = tbSeq << 1;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2 val_data[reg_size];

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + wid;
    if (iter_offset < total_count) {
        int iteration_stride = (blockDim.x >> 5) * gridDim.x;

        int seq_id = iter_offset % num_seq;
        int seq_id4 = seq_id >> 1;
        float max_val = minus_infinity;

        int mask_offset = (iter_offset / (heads * num_seq)) * (sequence_length);
        /**********
            [1 0 0 0]      [0 -inf -inf -inf]
            [1 1 0 0]      [0  0   -inf -inf]
            [1 1 1 0]      [0  0   0    -inf]
            [1 1 1 1]      [0  0   0       0]
        **********/

        for (int i = 0; i < reg_size; i++) {
            int data_id = i * (WARP_SIZE << 1) + (lane << 1);
            if ((!triangular || ((data_id >> 1) <= seq_id4)) && data_id < sequence_length) {
                if ((sequence_length - data_id) >= 2) {
                    val_data[i].x = shared_soft[wid * 1000 + data_id];
                    val_data[i].y = (!triangular || ((data_id + 1) <= seq_id))
                                        ? shared_soft[wid * 1000 + data_id + 1]
                                        : minus_infinity;

                    if (mask && !triangular && recompute) {
                        val_data[i].x += mask[data_id + mask_offset];
                        val_data[i].y += mask[data_id + mask_offset + 1];
                    }
                } else {
                    val_data[i].x = shared_soft[wid * 1000 + data_id];
                    val_data[i].y = minus_infinity;

                    if (mask && !triangular && recompute) {
                        val_data[i].x += mask[data_id + mask_offset];
                    }
                }
                max_val = (val_data[i].x > max_val ? val_data[i].x : max_val);
                max_val = (val_data[i].y > max_val ? val_data[i].y : max_val);
            } else {
                val_data[i].x = minus_infinity;
                val_data[i].y = minus_infinity;
            }
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        __shared__ float partialSum[MAX_WARP_NUM];

        if (Reduce_Threads > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            b.sync();

            if (lane < warp_num) max_val = partialSum[lane];

            int iters = warp_num;
            if (Reduce_Threads < iteration_stride) iters /= (iteration_stride / Reduce_Threads);

            for (int i = 1; i < iters; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }

        float sum = 0;
        for (int i = 0; i < reg_size; i++) {
            val_data[i].x = __expf(val_data[i].x - max_val);
            val_data[i].y = __expf(val_data[i].y - max_val);

            sum += (val_data[i].x + val_data[i].y);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum += g.shfl_xor(sum, i);

        if (Reduce_Threads > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            b.sync();

            if (lane < warp_num) sum = partialSum[lane];

            int iters = warp_num;
            if (Reduce_Threads < iteration_stride) iters /= (iteration_stride / Reduce_Threads);

            for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

            sum = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }
        sum += 1e-6;
        for (int i = 0; i < reg_size; i++) {
            int data_id = i * (WARP_SIZE << 1) + (lane << 1);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 2) {
                    val_data[i].x /= sum;
                    val_data[i].y /= sum;
                } else {
                    val_data[i].x /= sum;
                    val_data[i].y = 0;
                }
                shared_soft1[wid * 1000 + (data_id >> 1)] = val_data[i];
            }
        }
    }
}

__device__ void attn_context(__half2* shared_soft1,
                             __half* prev_value,
                             __half* new_value,
                             __half* merged_value,
                             bool merging,
                             __half* output,
                             int value_length,
                             int num_seq,
                             int hidden,
                             int head_size,
                             int total_count)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    __half2* output_cast = reinterpret_cast<__half2*>(output);

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* value_cast = reinterpret_cast<__half2*>(prev_value);
    __half2* new_value_cast;
    if (new_value) new_value_cast = reinterpret_cast<__half2*>(new_value);
    __half2* merged_value_cast;
    if (merging) merged_value_cast = reinterpret_cast<__half2*>(merged_value);

    int offset = (blockIdx.x * warp_num + wid) / num_seq;
    int value_size = total_count / num_seq;
    int unique_id = (blockIdx.x * warp_num + wid) % num_seq;
    if (offset < value_size) {
        __half2 vals_f[4];
        int wid_iter = 0;
        float2 sum[attn_warps << 1];
#pragma unroll
        for (int p = 0; p < (attn_warps << 1); p++) {
            sum[p].x = 0;
            sum[p].y = 0;
        }
        offset = (offset * head_size);
        int merge_offset = offset + lane;
        while (wid_iter < value_length) {
            {
                __half2 val_h[2];
                val_h[0] = shared_soft1[wid * 1024 + (wid_iter >> 1)];
                val_h[1] = shared_soft1[wid * 1024 + (wid_iter >> 1) + 1];

                __half* inp_data[2];
                inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                vals_f[0] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                vals_f[1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                vals_f[2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                vals_f[3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
            }
            int row = lane;
            int offset1 = lane + offset;
            int merge_offset1 = merge_offset;
            int iter = 0;
            while (row < head_size) {
                __half2 weight_h[4];
                weight_h[0] = value_cast[offset1];
                weight_h[1] = ((wid_iter + 1) < value_length ? value_cast[hidden + offset1]
                                                             : __float2half2_rn(0.f));
                weight_h[2] = ((wid_iter + 2) < value_length ? value_cast[(hidden << 1) + offset1]
                                                             : __float2half2_rn(0.f));
                weight_h[3] =
                    ((wid_iter + 3) < value_length ? value_cast[((hidden << 1) + hidden) + offset1]
                                                   : __float2half2_rn(0.f));
                if (merging && unique_id == 0) {
                    merged_value_cast[merge_offset1] = weight_h[0];
                    if ((wid_iter + 1) < value_length)
                        merged_value_cast[hidden + merge_offset1] = weight_h[1];
                    if ((wid_iter + 2) < value_length)
                        merged_value_cast[(hidden << 1) + merge_offset1] = weight_h[2];
                    if ((wid_iter + 3) < value_length)
                        merged_value_cast[((hidden << 1) + hidden) + merge_offset1] = weight_h[3];
                }
                {
                    float2 mul[4];
                    mul[0] = __half22float2(vals_f[0] * weight_h[0]);
                    mul[1] = __half22float2(vals_f[1] * weight_h[1]);
                    mul[2] = __half22float2(vals_f[2] * weight_h[2]);
                    mul[3] = __half22float2(vals_f[3] * weight_h[3]);

                    sum[iter].x += mul[0].x + mul[1].x + mul[2].x + mul[3].x;
                    sum[iter].y += mul[0].y + mul[1].y + mul[2].y + mul[3].y;
                }
                row += (WARP_SIZE);
                offset1 += (WARP_SIZE);
                merge_offset1 += WARP_SIZE;
                iter++;
            }
            wid_iter += 4;
            offset += (hidden * 4);
            merge_offset += (hidden * 4);
        }

        if (new_value) {
            int row = lane;
            int merge_offset = ((blockIdx.x * warp_num + wid) / num_seq) * head_size + lane +
                               (value_length * hidden);
            __half2 val_h;
            val_h = shared_soft1[wid * 1024 + (value_length >> 1)];
            __half* inp_data;
            inp_data = reinterpret_cast<__half*>(&val_h);
            vals_f[0] = __halves2half2(inp_data[value_length % 2], inp_data[value_length % 2]);
            int p = 0;
            int offset1 = ((blockIdx.x * warp_num + wid) / num_seq) * (head_size) + lane;
            while (row < head_size) {
                __half2 new_value_data = new_value_cast[offset1];
                float2 mul = __half22float2(vals_f[0] * new_value_data);
                sum[p].x += mul.x;
                sum[p].y += mul.y;
                if (merging && unique_id == 0) merged_value_cast[merge_offset] = new_value_data;
                row += WARP_SIZE;
                offset1 += WARP_SIZE;
                merge_offset += WARP_SIZE;
                p++;
            }
        }
        int offset1 = ((blockIdx.x * warp_num + wid));
        if (offset1 < total_count) {
            int p = 0;
            int row = lane;
            offset1 = offset1 * head_size + lane;
            while (row < head_size) {
                output_cast[offset1] = __float22half2_rn(sum[p]);
                row += WARP_SIZE;
                offset1 += WARP_SIZE;
                p++;
            }
        }
    }
}

__device__ void attn_context(float2* shared_soft1,
                             float* prev_value,
                             float* new_value,
                             float* merged_value,
                             bool merging,
                             float* output,
                             int value_length,
                             int num_seq,
                             int hidden,
                             int head_size,
                             int total_count)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    float2* output_cast = reinterpret_cast<float2*>(output);

    float2 ZERO_f2;
    ZERO_f2.x = ZERO_f2.y = 0.f;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2* value_cast = reinterpret_cast<float2*>(prev_value);
    float2* new_value_cast;
    if (new_value) new_value_cast = reinterpret_cast<float2*>(new_value);
    float2* merged_value_cast;
    if (merging) merged_value_cast = reinterpret_cast<float2*>(merged_value);

    int offset = (blockIdx.x * warp_num + wid) / num_seq;
    int value_size = total_count / num_seq;
    int unique_id = (blockIdx.x * warp_num + wid) % num_seq;
    if (offset < value_size) {
        float2 val_data;
        int wid_iter = 0;
        float2 sum[8];
#pragma unroll
        for (int p = 0; p < 8; p++) {
            sum[p].x = 0;
            sum[p].y = 0;
        }
        offset = (offset * head_size);
        int merge_offset = offset + lane;
        while (wid_iter < value_length) {
            {
                val_data = shared_soft1[wid * 1000 + (wid_iter >> 1)];
            }
            int row = lane;
            int offset1 = lane + offset;
            int merge_offset1 = merge_offset;
            int iter = 0;
            while (row < head_size) {
                float2 weight[2];
                weight[0] = value_cast[offset1];
                weight[1] =
                    ((wid_iter + 1) < value_length ? value_cast[hidden + offset1] : ZERO_f2);
                if (merging && unique_id == 0) {
                    merged_value_cast[merge_offset1] = weight[0];
                    if ((wid_iter + 1) < value_length)
                        merged_value_cast[hidden + merge_offset1] = weight[1];
                }
                float2 mul[2];
                {
                    mul[0].x = val_data.x * weight[0].x;
                    mul[0].y = val_data.x * weight[0].y;
                    mul[1].x = val_data.y * weight[1].x;
                    mul[1].y = val_data.y * weight[1].y;

                    sum[iter].x += mul[0].x + mul[1].x;
                    sum[iter].y += mul[0].y + mul[1].y;
                }
                row += (WARP_SIZE);
                offset1 += (WARP_SIZE);
                merge_offset1 += WARP_SIZE;
                iter++;
            }
            wid_iter += 2;
            offset += (hidden * 2);
            merge_offset += (hidden * 2);
        }

        if (new_value) {
            int row = lane;
            int merge_offset = ((blockIdx.x * warp_num + wid) / num_seq) * head_size + lane +
                               (value_length * hidden);
            val_data = shared_soft1[wid * 1000 + (value_length >> 1)];

            int p = 0;
            int offset1 = ((blockIdx.x * warp_num + wid) / num_seq) * (head_size) + lane;
            while (row < head_size) {
                float2 new_value_data = new_value_cast[offset1];
                float2 mul;

                mul.x = ((value_length % 2) ? val_data.y : val_data.x) * new_value_data.x;
                mul.y = ((value_length % 2) ? val_data.y : val_data.x) * new_value_data.y;

                sum[p].x += mul.x;
                sum[p].y += mul.y;
                if (merging && unique_id == 0) merged_value_cast[merge_offset] = new_value_data;
                row += WARP_SIZE;
                offset1 += WARP_SIZE;
                merge_offset += WARP_SIZE;
                p++;
            }
        }
        int offset1 = ((blockIdx.x * warp_num + wid));
        if (offset1 < total_count) {
            int p = 0;
            int row = lane;
            offset1 = offset1 * head_size + lane;
            while (row < head_size) {
                output_cast[offset1] = sum[p];
                row += WARP_SIZE;
                offset1 += WARP_SIZE;
                p++;
            }
        }
    }
}

template <int tbSize, int tbSeq>
__global__ void attn_softmax_context(__half* output,
                                     __half* query,
                                     __half* key,
                                     __half* new_key,
                                     __half* mask,
                                     float norm_factor,
                                     __half* key_merged,
                                     __half* prev_value,
                                     __half* new_value,
                                     __half* merged_value,
                                     bool merging,
                                     bool triangular,
                                     bool recompute,
                                     int total_count,
                                     int heads,
                                     int head_size,
                                     int value_length,
                                     int seq_length,
                                     int num_seq,
                                     float scale)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);

    if (iter_offset < total_count) {
        __shared__ __half2 shared_soft1[attn_warps * (1024 + 1)];
        int hidden = heads * head_size;
        {
            __shared__ __half shared_soft[attn_warps * (1024 + 1)];
            // Attntion_Score
            attn_score(shared_soft,
                       query,
                       key,
                       new_key,
                       key_merged,
                       merging,
                       norm_factor,
                       head_size,
                       total_count,
                       num_seq,
                       hidden,
                       value_length);
            b.sync();
            attn_softmax<tbSeq>(shared_soft,
                                shared_soft1,
                                mask,
                                heads,
                                total_count,
                                num_seq,
                                seq_length,
                                triangular,
                                recompute);
            b.sync();
        }
        // Attention_Context
        attn_context(shared_soft1,
                     prev_value,
                     new_value,
                     merged_value,
                     merging,
                     output,
                     value_length,
                     num_seq,
                     hidden,
                     head_size,
                     total_count);
    }
#endif
}

template <int tbSize, int tbSeq>
__global__ void attn_softmax_context(float* output,
                                     float* query,
                                     float* key,
                                     float* new_key,
                                     float* mask,
                                     float norm_factor,
                                     float* key_merged,
                                     float* prev_value,
                                     float* new_value,
                                     float* merged_value,
                                     bool merging,
                                     bool triangular,
                                     bool recompute,
                                     int total_count,
                                     int heads,
                                     int head_size,
                                     int value_length,
                                     int seq_length,
                                     int num_seq,
                                     float scale)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);

    // if (iter_offset < total_count)
    {
        __shared__ float2 shared_soft1[4004];
        int hidden = heads * head_size;
        {
            __shared__ float shared_soft[4004];
            // Attntion_Score
            attn_score(shared_soft,
                       query,
                       key,
                       new_key,
                       key_merged,
                       merging,
                       norm_factor,
                       head_size,
                       total_count,
                       num_seq,
                       hidden,
                       value_length);
            b.sync();
            attn_softmax<tbSeq>(shared_soft,
                                shared_soft1,
                                mask,
                                heads,
                                total_count,
                                num_seq,
                                seq_length,
                                triangular,
                                recompute);
            b.sync();
        }
        // Attention_Context
        attn_context(shared_soft1,
                     prev_value,
                     new_value,
                     merged_value,
                     merging,
                     output,
                     value_length,
                     num_seq,
                     hidden,
                     head_size,
                     total_count);
    }
#endif
}

template <typename T>
void launch_attn_softmax_context(T* out,
                                 T* query,
                                 T* key,
                                 T* new_key,
                                 T* mask,
                                 float norm_factor,
                                 T* key_merged,
                                 T* prev_value,
                                 T* new_value,
                                 T* merged_value,
                                 bool merging,
                                 bool triangular,
                                 bool recompute,
                                 int batch_size,
                                 int heads,
                                 int head_size,
                                 int value_length,
                                 int num_seq,
                                 int sequence_length,
                                 float scale,
                                 cudaStream_t stream)
{
    int total_count = batch_size * heads * num_seq;

    dim3 grid_dim((total_count - 1) / attn_warps + 1);
    dim3 block_dim(Attn_Threads);
    if (sequence_length <= 128)
        attn_softmax_context<32, 1><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                        query,
                                                                        key,
                                                                        new_key,
                                                                        mask,
                                                                        norm_factor,
                                                                        key_merged,
                                                                        prev_value,
                                                                        new_value,
                                                                        merged_value,
                                                                        merging,
                                                                        triangular,
                                                                        recompute,
                                                                        total_count,
                                                                        heads,
                                                                        head_size / 2,
                                                                        value_length,
                                                                        sequence_length,
                                                                        num_seq,
                                                                        scale);
    else if (sequence_length <= 256)
        attn_softmax_context<32, 2><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                        query,
                                                                        key,
                                                                        new_key,
                                                                        mask,
                                                                        norm_factor,
                                                                        key_merged,
                                                                        prev_value,
                                                                        new_value,
                                                                        merged_value,
                                                                        merging,
                                                                        triangular,
                                                                        recompute,
                                                                        total_count,
                                                                        heads,
                                                                        head_size / 2,
                                                                        value_length,
                                                                        sequence_length,
                                                                        num_seq,
                                                                        scale);
    else if (sequence_length <= 512)
        attn_softmax_context<32, 4><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                        query,
                                                                        key,
                                                                        new_key,
                                                                        mask,
                                                                        norm_factor,
                                                                        key_merged,
                                                                        prev_value,
                                                                        new_value,
                                                                        merged_value,
                                                                        merging,
                                                                        triangular,
                                                                        recompute,
                                                                        total_count,
                                                                        heads,
                                                                        head_size / 2,
                                                                        value_length,
                                                                        sequence_length,
                                                                        num_seq,
                                                                        scale);
    else if (sequence_length <= 1024)
        attn_softmax_context<32, 8><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                        query,
                                                                        key,
                                                                        new_key,
                                                                        mask,
                                                                        norm_factor,
                                                                        key_merged,
                                                                        prev_value,
                                                                        new_value,
                                                                        merged_value,
                                                                        merging,
                                                                        triangular,
                                                                        recompute,
                                                                        total_count,
                                                                        heads,
                                                                        head_size / 2,
                                                                        value_length,
                                                                        sequence_length,
                                                                        num_seq,
                                                                        scale);
    else if (sequence_length <= 2048)
        attn_softmax_context<32, 16><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                         query,
                                                                         key,
                                                                         new_key,
                                                                         mask,
                                                                         norm_factor,
                                                                         key_merged,
                                                                         prev_value,
                                                                         new_value,
                                                                         merged_value,
                                                                         merging,
                                                                         triangular,
                                                                         recompute,
                                                                         total_count,
                                                                         heads,
                                                                         head_size / 2,
                                                                         value_length,
                                                                         sequence_length,
                                                                         num_seq,
                                                                         scale);
    else if (sequence_length <= 4096)
        attn_softmax_context<32, 32><<<grid_dim, block_dim, 0, stream>>>(out,
                                                                         query,
                                                                         key,
                                                                         new_key,
                                                                         mask,
                                                                         norm_factor,
                                                                         key_merged,
                                                                         prev_value,
                                                                         new_value,
                                                                         merged_value,
                                                                         merging,
                                                                         triangular,
                                                                         recompute,
                                                                         total_count,
                                                                         heads,
                                                                         head_size / 2,
                                                                         value_length,
                                                                         sequence_length,
                                                                         num_seq,
                                                                         scale);
    else
        throw std::runtime_error(
            "Unsupport Seq_Length! Check the restriction of the max_threads and "
            "max_thread_iterations!");

    CUDA_CHECK_ERROR();
}

template void launch_attn_softmax_context(float* out,
                                          float* query,
                                          float* key,
                                          float* new_key,
                                          float* mask,
                                          float norm_factor,
                                          float* key_merged,
                                          float* prev_value,
                                          float* new_value,
                                          float* merged_value,
                                          bool merging,
                                          bool triangular,
                                          bool recompute,
                                          int batch_size,
                                          int heads,
                                          int head_size,
                                          int value_length,
                                          int num_seq,
                                          int sequence_length,
                                          float scale,
                                          cudaStream_t stream);

template void launch_attn_softmax_context(__half* out,
                                          __half* query,
                                          __half* key,
                                          __half* new_key,
                                          __half* mask,
                                          float norm_factor,
                                          __half* key_merged,
                                          __half* prev_value,
                                          __half* new_value,
                                          __half* merged_value,
                                          bool merging,
                                          bool triangular,
                                          bool recompute,
                                          int batch_size,
                                          int heads,
                                          int head_size,
                                          int value_length,
                                          int num_seq,
                                          int sequence_length,
                                          float scale,
                                          cudaStream_t stream);
