#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#define MAX_WARP_NUM 32
#define WARP_SIZE 32
#define SMs 80

#define MAX_REGISTERS 256
template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            bool triangular,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            float scale,
                            cudaStream_t stream);
template <typename T>
void launch_wmma_tiled_gemm(T* output,
                            T* vals,
                            T* weight,
                            T* bias,
                            unsigned batch_size,
                            unsigned hidden_dim,
                            unsigned output_size,
                            bool add_gelu,
                            cudaStream_t stream);
template <typename T>
void launch_attn_softmax_context(T* out,
                                 T* query,
                                 T* key,
                                 T* new_key,
                                 T* attn_mask,
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
                                 cudaStream_t stream);
// Custom bias add
template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    int hidden_dim,
                                    int heads,
                                    cudaStream_t stream,
                                    int trans_count);
// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);
template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream);

template <typename T>
void launch_bias_residual(T* input,
                          const T* residual,
                          const T* bias,
                          int size,
                          int intermediate_size,
                          cudaStream_t stream);

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);
template <typename T>
void launch_residual_layer_norm(T* norm,
                                T* res_add,
                                T* vals,
                                T* residual,
                                const T* bias,
                                const T* gamma,
                                const T* beta,
                                float epsilon,
                                int batch_size,
                                int hidden_dim,
                                cudaStream_t stream);
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
                                    cudaStream_t stream);
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
                                       cudaStream_t stream);
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
                                       cudaStream_t stream);
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
                                         cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
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
                                    cudaStream_t stream);

template <typename T>
void launch_tiled_gemm_kernel_gelu(T* output,
                                   const T* vals,
                                   const T* weight,
                                   const T* bias,
                                   int hidden_dim,
                                   int input_size,
                                   int output_size,
                                   cudaStream_t stream);
