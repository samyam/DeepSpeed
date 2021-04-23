

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "context.h"
#include "cublas_wrappers.h"
#include "custom_cuda_layers.h"

std::array<int, 3> gemm_algos = std::array<int, 3>({99, 99, 99});

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores, at::Tensor& attn_mask, int padding)
{
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);
    int seq_len = attn_scores_c.size(2);
    int soft_len = attn_scores_c.size(3) - padding;
    int heads = attn_scores_c.size(1);

    launch_attn_softmax_v2((T*)attn_scores_c.data_ptr(),
                           (T*)attn_mask.data_ptr(),
                           (seq_len == soft_len),
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           1.0,
                           at::cuda::getCurrentCUDAStream());

    return attn_scores_c;
}

template <typename T>
void attention_unfused(at::Tensor& prev_key_cont,
                       at::Tensor& query_cont,
                       at::Tensor& attn_mask,
                       at::Tensor& prev_value_cont,
                       at::Tensor& output,
                       int& bsz,
                       int& seq_len,
                       int& soft_len,
                       int& heads,
                       float& norm_factor)
{
    auto options = at::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    float alpha = 1.0 / norm_factor;
    float gemm_beta = 0.0;
    auto attn_score = at::zeros({bsz, heads, seq_len, soft_len}, options);
    int k = prev_value_cont.size(2) / heads;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont.data_ptr(),
                                (T*)query_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    attn_score = ds_softmax<T>(attn_score, attn_mask, 0);
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                (T*)output.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template <typename T>
std::vector<at::Tensor> ds_softmax_context(at::Tensor& query,
                                           at::Tensor& prev_key,
                                           at::Tensor& new_key,
                                           at::Tensor& attn_mask,
                                           at::Tensor& prev_value,
                                           at::Tensor& new_value,
                                           int heads,
                                           float norm_factor,
                                           bool merging)
{
    auto query_cont = query.contiguous();
    auto prev_key_cont = prev_key.contiguous();
    auto new_key_cont = new_key.contiguous();
    auto prev_value_cont = prev_value.contiguous();
    auto new_value_cont = new_value.contiguous();

    int new_size = (new_value.sizes().size() > 1 ? new_value.size(1) : 0);

    // Attn_Score [ batch Head Sequence-length Softmax-length]

    int bsz = query_cont.size(0);
    int seq_len = query_cont.size(1);
    int soft_len = prev_value.size(1) + new_size;

    auto options = at::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto merged_value =
        (merging
             ? at::empty({prev_value.size(0), (prev_value.size(1) + new_size), prev_value.size(2)},
                         options)
             : prev_value);
    auto merged_key =
        (merging ? at::empty({prev_key.size(0), (prev_key.size(1) + new_size), prev_key.size(2)},
                             options)
                 : prev_key);
    auto output =
        at::empty({prev_value.size(0), heads, seq_len, prev_value.size(2) / heads}, options);
    if (seq_len >= 32) {
        attention_unfused<T>(prev_key_cont,
                             query_cont,
                             attn_mask,
                             prev_value_cont,
                             output,
                             bsz,
                             seq_len,
                             soft_len,
                             heads,
                             norm_factor);
    } else {
        launch_attn_softmax_context((T*)output.data_ptr(),
                                    (T*)query_cont.data_ptr(),
                                    (T*)prev_key_cont.data_ptr(),
                                    (T*)(new_size > 0 ? new_key_cont.data_ptr() : nullptr),
                                    (T*)(attn_mask.data_ptr()),
                                    norm_factor,
                                    (T*)(merging ? merged_key.data_ptr() : nullptr),
                                    (T*)prev_value_cont.data_ptr(),
                                    (T*)(new_size > 0 ? new_value_cont.data_ptr() : nullptr),
                                    (T*)(merging ? merged_value.data_ptr() : nullptr),
                                    merging,
                                    (seq_len == soft_len),  // Triangular
                                    (new_size == 0),        // recompute
                                    bsz,
                                    heads,
                                    prev_value.size(2) / heads,
                                    prev_value.size(1),
                                    seq_len,
                                    soft_len,
                                    1.0,
                                    at::cuda::getCurrentCUDAStream());
    }
    return {output, merged_key, merged_value};
}

template <typename T>
at::Tensor ds_bias_gelu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_gelu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_residual(at::Tensor& input, at::Tensor& residual, at::Tensor& bias)
{
    auto input_cont = input.contiguous();
    auto residual_cont = residual.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);

    launch_bias_residual((T*)input_cont.data_ptr(),
                         (T*)residual_cont.data_ptr(),
                         (T*)bias.data_ptr(),
                         bsz,
                         input_cont.size(2),
                         Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_layernorm(at::Tensor& input_cont, at::Tensor& gamma, at::Tensor& betta, float epsilon)
{
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);
    launch_layer_norm((T*)inp_norm.data_ptr(),
                      (T*)input_cont.data_ptr(),
                      (T*)gamma.data_ptr(),
                      (T*)betta.data_ptr(),
                      epsilon,
                      bsz,
                      input_cont.size(2),
                      Context::Instance().GetCurrentStream());
    return inp_norm;
}

template <typename T>
void qkv_unfused_cublas(at::Tensor& output,
                        at::Tensor& input,
                        at::Tensor& weight,
                        at::Tensor& bias,
                        at::Tensor& gamma,
                        at::Tensor& beta,
                        const float epsilon)
{
    auto inp_norm = ds_layernorm<T>(input, gamma, beta, epsilon);
    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    int bsz = input.size(0) * input.size(1);
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)inp_norm.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    launch_bias_add((T*)output.data_ptr(),
                    (T*)bias.data_ptr(),
                    weight.size(1),
                    bsz,
                    Context::Instance().GetCurrentStream());
}

template <typename T>
at::Tensor ds_qkv_gemm(at::Tensor& input,
                       at::Tensor& weight,
                       at::Tensor& bias,
                       at::Tensor& gamma,
                       at::Tensor& beta,
                       const float epsilon)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);
    if (bsz >= 16) {
        qkv_unfused_cublas<T>(output, input_cont, weight, bias, gamma, beta, epsilon);
    } else {
        // computing the blocking across K dimension
        int out_blocks = (weight.size(1) - 1) / 64 + 1;
        out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
        int br2 = (int)log2(out_blocks);
        out_blocks = (int)pow(2.0, (float)br2);
        if (input_cont.size(0) == 1 && out_blocks == 1) {
            launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                           (T*)input_cont.data_ptr(),
                                           (T*)weight.data_ptr(),
                                           (T*)bias.data_ptr(),
                                           (T*)gamma.data_ptr(),
                                           (T*)beta.data_ptr(),
                                           epsilon,
                                           input_cont.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream());
        } else {
#if !defined(__CUBLAS_UNFUSED__)
            auto inp_norm = ds_layernorm<T>(input_cont, gamma, beta, epsilon);
            auto block_sums = at::empty(
                {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
            launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                              (T*)inp_norm.data_ptr(),
                                              (T*)weight.data_ptr(),
                                              (T*)bias.data_ptr(),
                                              (T*)block_sums.data_ptr(),
                                              input_cont.size(2),
                                              bsz,
                                              weight.size(1),
                                              false,
                                              Context::Instance().GetCurrentStream());
#else

            qkv_unfused_cublas<T>(output, input_cont, weight, bias, gamma, beta, epsilon);
#endif
        }
    }
    return output;
}

template <typename T>
at::Tensor ds_qkv_gemm_int8(at::Tensor& input,
                            at::Tensor& weight,
                            at::Tensor& bias,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            const float epsilon,
                            at::Tensor& q_scale,
                            int groups)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    // computing the blocking across K dimension
    int out_blocks = (weight.size(1) - 1) / 128 + 1;
    out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
    int br2 = (int)log2(out_blocks);
    out_blocks = (int)pow(2.0, (float)br2);
    if (input_cont.size(0) == 1 && out_blocks == 1) {
        launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                       (T*)input_cont.data_ptr(),
                                       (int8_t*)weight.data_ptr(),
                                       (T*)bias.data_ptr(),
                                       (T*)gamma.data_ptr(),
                                       (T*)beta.data_ptr(),
                                       epsilon,
                                       input_cont.size(2),
                                       bsz,
                                       weight.size(1),
                                       (float*)q_scale.data_ptr(),
                                       groups,
                                       Context::Instance().GetCurrentStream());
    } else {
        auto inp_norm = ds_layernorm<T>(input_cont, gamma, beta, epsilon);
        auto block_sums = at::empty(
            {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
        launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                          (T*)inp_norm.data_ptr(),
                                          (int8_t*)weight.data_ptr(),
                                          (T*)bias.data_ptr(),
                                          input_cont.size(2),
                                          bsz,
                                          weight.size(1),
                                          (float*)q_scale.data_ptr(),
                                          groups,
                                          0,
                                          (T*)block_sums.data_ptr(),
                                          false,
                                          Context::Instance().GetCurrentStream());
    }
    return output;
}

template <typename T>
at::Tensor ds_linear_layer(at::Tensor& input, at::Tensor& weight, at::Tensor& bias)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);
    if (bsz > 1) {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
                       bsz,
                       input_cont.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input_cont.data_ptr(),
                       (T*)output.data_ptr(),
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
    } else {
        launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                       (T*)input_cont.data_ptr(),
                                       (T*)weight.data_ptr(),
                                       (T*)bias.data_ptr(),
                                       input_cont.size(2),
                                       bsz,
                                       weight.size(1),
                                       Context::Instance().GetCurrentStream());
    }
    return output;
}

template <typename T>
at::Tensor ds_linear_layer_int8(at::Tensor& input,
                                at::Tensor& weight,
                                at::Tensor& bias,
                                at::Tensor& q_scale,
                                int groups)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);

    // computing the blocking across K dimension
    int out_blocks = (weight.size(1) - 1) / 128 + 1;
    out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
    int br2 = (int)log2(out_blocks);
    out_blocks = (int)pow(2.0, (float)br2);
    if (input_cont.size(0) == 1 && out_blocks == 1) {
        launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                       (T*)input_cont.data_ptr(),
                                       (int8_t*)weight.data_ptr(),
                                       (T*)bias.data_ptr(),
                                       input_cont.size(2),
                                       bsz,
                                       weight.size(1),
                                       (float*)q_scale.data_ptr(),
                                       groups,
                                       1,
                                       Context::Instance().GetCurrentStream());
    } else {
        auto block_sums = at::empty(
            {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
        launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                          (T*)input_cont.data_ptr(),
                                          (int8_t*)weight.data_ptr(),
                                          (T*)bias.data_ptr(),
                                          input_cont.size(2),
                                          bsz,
                                          weight.size(1),
                                          (float*)q_scale.data_ptr(),
                                          groups,
                                          1,
                                          (T*)block_sums.data_ptr(),
                                          false,
                                          Context::Instance().GetCurrentStream());
    }
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input, at::Tensor& weight)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);
    if (bsz >= 16) {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
                       bsz,
                       input_cont.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input_cont.data_ptr(),
                       (T*)output.data_ptr(),
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // computing the blocking across K dimension
        int out_blocks = (weight.size(1) - 1) / 64 + 1;
        out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
        int br2 = (int)log2(out_blocks);
        out_blocks = (int)pow(2.0, (float)br2);

        // selecting the high-performance kernels for vector-matrix multiplication
        if (input_cont.size(0) == 1 && out_blocks == 1)
            launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                           (T*)input_cont.data_ptr(),
                                           (T*)weight.data_ptr(),
                                           (T*)nullptr,
                                           input_cont.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream());
        else {
#if !defined(__CUBLAS_UNFUSED__)
            // v2 gemm kernels partitions the weight across output and K dimension
            // to allocate more blocks and saturate the GPU memory bandwidth
            auto block_sums = at::empty(
                {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
            launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                              (T*)input_cont.data_ptr(),
                                              (T*)weight.data_ptr(),
                                              (T*)nullptr,
                                              (T*)block_sums.data_ptr(),
                                              input_cont.size(2),
                                              bsz,
                                              weight.size(1),
                                              false,
                                              Context::Instance().GetCurrentStream());
#else
            float alpha = (T)1.0;
            float gemm_beta = (T)0.0;
            cublasSetStream(Context::Instance().GetCublasHandle(),
                            Context::Instance().GetCurrentStream());
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight.size(1),
                           bsz,
                           input_cont.size(2),
                           &alpha,
                           &gemm_beta,
                           (T*)weight.data_ptr(),
                           (T*)input_cont.data_ptr(),
                           (T*)output.data_ptr(),
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
        }
    }
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul_int8(at::Tensor& input,
                                 at::Tensor& weight,
                                 at::Tensor& q_scale,
                                 int groups,
                                 int merge_count)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);

    // computing the blocking across K dimension
    int out_blocks = (weight.size(1) - 1) / 128 + 1;
    out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
    int br2 = (int)log2(out_blocks);
    out_blocks = (int)pow(2.0, (float)br2);
    if (input_cont.size(0) == 1 && out_blocks == 1) {
        launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                       (T*)input_cont.data_ptr(),
                                       (int8_t*)weight.data_ptr(),
                                       (T*)nullptr,
                                       input_cont.size(2),
                                       bsz,
                                       weight.size(1),
                                       (float*)q_scale.data_ptr(),
                                       groups,
                                       merge_count,
                                       Context::Instance().GetCurrentStream());
    } else {
        auto block_sums = at::empty(
            {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
        launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                          (T*)input_cont.data_ptr(),
                                          (int8_t*)weight.data_ptr(),
                                          (T*)nullptr,
                                          input_cont.size(2),
                                          bsz,
                                          weight.size(1),
                                          (float*)q_scale.data_ptr(),
                                          groups,
                                          merge_count,
                                          (T*)block_sums.data_ptr(),
                                          false,
                                          Context::Instance().GetCurrentStream());
    }
    return output;
}

template <typename T>
void mlp_unfused_cublas(at::Tensor& output,
                        at::Tensor& residual_add,
                        at::Tensor& input,
                        at::Tensor& residual,
                        at::Tensor& input_bias,
                        at::Tensor& weight,
                        at::Tensor& bias,
                        at::Tensor& gamma,
                        at::Tensor& beta,
                        const float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    auto inp_norm = at::empty_like(input);

    launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                               (T*)residual_add.data_ptr(),
                               (T*)input.data_ptr(),
                               (T*)residual.data_ptr(),
                               (T*)input_bias.data_ptr(),
                               (T*)gamma.data_ptr(),
                               (T*)beta.data_ptr(),
                               epsilon,
                               bsz,
                               input.size(2),
                               Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)inp_norm.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());
}
template <typename T>
std::vector<at::Tensor> ds_mlp_gemm(at::Tensor& input,
                                    at::Tensor& residual,
                                    at::Tensor& input_bias,
                                    at::Tensor& weight,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    auto residual_add = at::empty_like(input_cont);
    int bsz = input_cont.size(0) * input_cont.size(1);
    if (bsz >= 16) {
        mlp_unfused_cublas<T>(
            output, residual_add, input, residual, input_bias, weight, bias, gamma, beta, epsilon);
    } else {
        // computing the blocking across K dimension
        int out_blocks = (weight.size(1) - 1) / 64 + 1;
        out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
        int br2 = (int)log2(out_blocks);
        out_blocks = (int)pow(2.0, (float)br2);

        if (input_cont.size(0) == 1 && out_blocks == 1) {
            launch_input_tiled_gemm_kernel_gelu((T*)output.data_ptr(),
                                                (T*)residual_add.data_ptr(),
                                                (T*)input_cont.data_ptr(),
                                                (T*)residual.data_ptr(),
                                                (T*)input_bias.data_ptr(),
                                                (T*)weight.data_ptr(),
                                                (T*)bias.data_ptr(),
                                                (T*)gamma.data_ptr(),
                                                (T*)beta.data_ptr(),
                                                epsilon,
                                                input_cont.size(2),
                                                bsz,
                                                weight.size(1),
                                                Context::Instance().GetCurrentStream());
        } else {
#if !defined(__CUBLAS_UNFUSED__)
            auto inp_norm = at::empty_like(input_cont);
            launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                                       (T*)residual_add.data_ptr(),
                                       (T*)input_cont.data_ptr(),
                                       (T*)residual.data_ptr(),
                                       (T*)input_bias.data_ptr(),
                                       (T*)gamma.data_ptr(),
                                       (T*)beta.data_ptr(),
                                       epsilon,
                                       bsz,
                                       input_cont.size(2),
                                       Context::Instance().GetCurrentStream());
            auto block_sums = at::empty(
                {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
            launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                              (T*)inp_norm.data_ptr(),
                                              (T*)weight.data_ptr(),
                                              (T*)bias.data_ptr(),
                                              (T*)block_sums.data_ptr(),
                                              input_cont.size(2),
                                              bsz,
                                              weight.size(1),
                                              true,
                                              Context::Instance().GetCurrentStream());
#else
            mlp_unfused_cublas<T>(output,
                                  residual_add,
                                  input,
                                  residual,
                                  input_bias,
                                  weight,
                                  bias,
                                  gamma,
                                  beta,
                                  epsilon);
#endif
        }
    }
    return {output, residual_add};
}

template <typename T>
std::vector<at::Tensor> ds_mlp_gemm_int8(at::Tensor& input,
                                         at::Tensor& residual,
                                         at::Tensor& input_bias,
                                         at::Tensor& weight,
                                         at::Tensor& bias,
                                         at::Tensor& gamma,
                                         at::Tensor& beta,
                                         const float epsilon,
                                         at::Tensor& q_scale,
                                         int groups)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    auto residual_add = at::empty_like(input_cont);
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);

    // computing the blocking across K dimension
    int out_blocks = (weight.size(1) - 1) / 128 + 1;
    out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
    int br2 = (int)log2(out_blocks);
    out_blocks = (int)pow(2.0, (float)br2);
    if (input_cont.size(0) == 1 && out_blocks == 1) {
        launch_input_tiled_gemm_kernel_gelu((T*)output.data_ptr(),
                                            (T*)residual_add.data_ptr(),
                                            (T*)input_cont.data_ptr(),
                                            (T*)residual.data_ptr(),
                                            (T*)input_bias.data_ptr(),
                                            (int8_t*)weight.data_ptr(),
                                            (T*)bias.data_ptr(),
                                            (T*)gamma.data_ptr(),
                                            (T*)beta.data_ptr(),
                                            epsilon,
                                            input_cont.size(2),
                                            bsz,
                                            weight.size(1),
                                            (float*)q_scale.data_ptr(),
                                            groups,
                                            Context::Instance().GetCurrentStream());
    } else {
        launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                                   (T*)residual_add.data_ptr(),
                                   (T*)input_cont.data_ptr(),
                                   (T*)residual.data_ptr(),
                                   (T*)input_bias.data_ptr(),
                                   (T*)gamma.data_ptr(),
                                   (T*)beta.data_ptr(),
                                   epsilon,
                                   bsz,
                                   input_cont.size(2),
                                   Context::Instance().GetCurrentStream());
        auto block_sums = at::empty(
            {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)}, options);
        launch_input_tiled_gemm_kernel_v2((T*)output.data_ptr(),
                                          (T*)inp_norm.data_ptr(),
                                          (int8_t*)weight.data_ptr(),
                                          (T*)bias.data_ptr(),
                                          input_cont.size(2),
                                          bsz,
                                          weight.size(1),
                                          (float*)q_scale.data_ptr(),
                                          groups,
                                          0,
                                          (T*)block_sums.data_ptr(),
                                          true,
                                          Context::Instance().GetCurrentStream());
    }
    return {output, residual_add};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def("softmax_fp16", &ds_softmax<__half>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def(
        "softmax_context_fp32", &ds_softmax_context<float>, "DeepSpeed attention with fp32 (CUDA)");
    m.def("softmax_context_fp16",
          &ds_softmax_context<__half>,
          "DeepSpeed attention with fp32 (CUDA)");
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_residual_fp32",
          &ds_bias_residual<float>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("bias_residual_fp16",
          &ds_bias_residual<__half>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("layer_norm_fp32", &ds_layernorm<float>, "DeepSpeed layer-norm with fp32 (CUDA)");
    m.def("layer_norm_fp16", &ds_layernorm<__half>, "DeepSpeed layer-norm with fp16 (CUDA)");
    m.def("qkv_gemm_fp32", &ds_qkv_gemm<float>, "DeepSpeed qkv gemm with fp32 (CUDA)");
    m.def("qkv_gemm_fp16", &ds_qkv_gemm<__half>, "DeepSpeed qkv gemm with fp16 (CUDA)");
    m.def("qkv_gemm_int8", &ds_qkv_gemm_int8<__half>, "DeepSpeed qkv gemm with int8 (CUDA)");
    m.def("mlp_gemm_fp32", &ds_mlp_gemm<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("mlp_gemm_fp16", &ds_mlp_gemm<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("mlp_gemm_int8", &ds_mlp_gemm_int8<__half>, "DeepSpeed mlp with int8 (CUDA)");
    m.def("vector_matmul_fp32", &ds_vector_matmul<float>, "DeepSpeed vector-MM with fp32 (CUDA)");
    m.def("vector_matmul_fp16", &ds_vector_matmul<__half>, "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("vector_matmul_int8",
          &ds_vector_matmul_int8<__half>,
          "DeepSpeed vector-MM with int8 (CUDA)");
    m.def("linear_layer_fp32", &ds_linear_layer<float>, "DeepSpeed linear_layer with fp32 (CUDA)");
    m.def("linear_layer_fp16", &ds_linear_layer<__half>, "DeepSpeed linear_layer with fp16 (CUDA)");
    m.def("linear_layer_int8",
          &ds_linear_layer_int8<__half>,
          "DeepSpeed linear_layer with int8 (CUDA)");
}
