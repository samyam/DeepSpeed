---
layout: single
title: "DeepSpeed Inference: Multi-GPU inference with customized inference kerenls and quantization support"
excerpt: ""
categories: news
new_post: false
date: 2021-03-16 00:00:00
---

While we can train advanced large-scale models now, using these trained models in the desired application scenarios is still challenging.  It requires powerful and efficient inference support, where existing inference solutions have their limitations. First, large models do not fit in a single GPU. To the best of our knowledge, well-known inference solutions, such as ONNX Runtime or NVIDIA TensorRT, do not support multi-GPU inference with a model partitioned across GPUs. Moreover, the new version of NVIDIA Triton-server can only support pipeline-parallelism for supporting large model inference, whereas we provide the flexibility of choosing between different types of parallelism for getting the best latency-throughput trade-off.

Second, even though trained models can be easily turned into inference using PyTorch directly, the inference performance is often far from optimal because efficient inference requires different parallelism considerations and GPU kernels with different optimizations than training. Although quantization has good potential to optimize inference performance, it is often difficult to quantize without losing accuracy and to serve quantized models efficiently on commodity hardware. To handle these challenges, we introduce DeepSpeed Inference, which seamlessly turns large-scale models trained in DeepSpeed into high-performance inference with inference-adapted parallelism, inference-optimized kernels, and flexible quantization support for training and inference.


## Multi-GPU Inference with Adaptive Parallelism

Parallelism is an effective approach to fit large models and reduce per-device memory consumption for both training and inference. However, simply applying training parallelism choices and degree to inference does not work well. The MP and PP configuration is normally set during the model training, apart from the data parallelism (DP), based on the memory footprint and computation style, and resource budget. On one hand, inference computation intrinsically requires less memory, so it can afford a larger partition per device. It helps reduce the degree of parallelism needed for model deployment. On the other hand, optimizing latency or meeting latency requirements is often a first-class citizen in inference while training optimizes throughput.

To obtain desired latency, DeepSpeed Inference automatically adapts MP as an effective approach to reduce model latency, and its parallelism degree is often determined first. With MP, we can split the mode and parallelize computational operations across multiple devices (GPUs) to reduce latency, but it reduces computation granularity and increases communication that may hurt throughput. Once the latency target has been met, DeepSpeed can apply pipeline parallelism to maximize the throughput. Overall, DeepSpeed Inference supports flexible adaptation of both parallelism approach and degree choices from training to inference, minimizing latency while saving deployment costs.


## Customized Inference Kernels for Boosted Compute Efficiency of Transformer Blocks

To achieve high compute efficiency, DeepSpeed-inference offers inference kernels tailored for Transformer blocks through operator fusion, taking model-parallelism for multi-GPU into account. The main difference between our kernel-fusion scheme and similar approaches is that we not only fuse element-wise operations (such as bias-add, residual, and activation function), but also merge the General matrix multiply (GeMM) operations with other operations. To do this, we design an efficient implementation for the vector-matrix or skinny matrix-matrix multiplication that allows us to fuse more operations at the reduction boundary of GeMM operations.

# Kernel-Fusion

We take two main policies for fusing operations: 1) keeping the access-pattern of inputs and outputs intact throughout the sequence of operations fused together; 2) fusing operations at each all-reduce boundary. The first policy ensures that different thread-blocks won’t encounter transferring data between Streaming-Multiprocessors (SMs). This is due to no straight-forward communication among SMs other than using the main memory which adds the block-synching overhead because of non-deterministic behavior of memory access. The reason behind the second policy is that we cannot continue the execution unless the partial results are reduced among the model-parallel GPUs.

![Inference-Kernel-Fusion](/assets/images/inference-kernel-fusion.png){: .align-center}

Figure 1: Transformer Layer with Megatron-style model-parallelism all-reduce components. The figure illustrates the parts of layer fused together with broken lines (width of line shows the fusion depth).

Figure 1 shows the different components of a Transformer layer, and the groups of operations considered for fusion in our inference optimization. We also consider the NVIDIA Megatron-LM style of parallelism that partitions attention (Attn) and feed-forward (FF) blocks across multiple GPUs. Thus, we include the two all-reduce operations that reduce the results among parallel GPUs after Attn and FF blocks. As Figure 1 shows, we fuse the operations inside a Transformer layer at four main regions:
1.	Input Layer-Norm plus Query, Key, and Value GeMMs and their bias adds.
2.	Transform plus Attention.
3.  Intermediate FF, Layer-Norm, Bias-add, Residual, and Gaussian Error Linear Unit (GELU).
4.	Bias-add plus Residual.

To fuse these operations, we exploit shared-memory as an intermediate cache for transferring data between reduction operations used in layer-norm and GeMM, and the element-wise operations. Moreover, we use the warp-level instructions to communicate data between threads when reducing partial computations. In addition, we use a new schedule for GeMM operations, which allows for fusing as many operations as needed for the third kernel-fusion. We also combine the GeMMs for the attention computation in the second kernel-fusion, by using an implicit matrix transformation in order to reduce the memory pressure. Compared to the unfused computation style using cuBLAS GeMM, we improve the performance by 1.5x, 2.9x. 3x, and 1.2x for all these kernel-fusions, respectively.

# GeMM Scheduling

GeMMs are the most important operations inside a Transformer, taking more than 60–70% of the runtime. Therefore, unless this operation is fully optimized, our kernel fusion won’t be effective. Hence, we explored multiple ways of GeMM scheduling for vector-matrix or skinny matrix-matrix and found the one which suits the most for the inference needs. The winner schedule is a variant of [Persistent-Threads](https://ieeexplore.ieee.org/document/6339596) (PT) approach, with the difference that it takes into account the hardware architecture as well as the memory-access pattern rather than keeping the data persistent on each thread.

![GeMM-Scheduling](/assets/images/inference-gemm-scheduling.png){: .align-center}

Figure 2. GeMM scheduling at different GPU architectural levels: threads-blocks, warps and CUDA threads. warps show different cooperative threads (32 threads), Lanes show the thread index at each warp.

Figure 2 depicts our GeMM scheduling for a skinny matrix-matrix multiplication. We partition the weights in two directions: 1) column-wise across the threads of the first Warp of all blocks, and 2) row-wise across the warps at each block. By assigning the columns to different blocks, we gain two important benefits: coalesced access on the weight matrix and good locality on the input matrix. Coalesced access is the result of consecutive memory access on the cooperative threads across different blocks. Moreover, we use the full bandwidth of L1 cache as the concurrent 32 threads access the same cache line, bringing 128 bytes of data. Furthermore, input buffer is accessed in the same pattern at all blocks, which improves the cache locality.

To perform the main GeMM operation, each portion of a column segment is mapped to different warps in a block. Then, each warp computes the local GeMM over its portion iteratively until reaching the last row of the matrix. Next, we use shared memory to transpose data to keep the summation reduction local to warps. This alleviates the thread-communication overhead, as each column’s data maps to the same warp.

After reducing the summations, we have the result of a column of input tiles at each warp. However, if we write the results in the same pattern, it would result in uncoalesced access for the output matrix. To handle this more efficiently, we once again transpose data with the help of shared memory. Finally, the first few lanes of each warp write back the result into the output matrix. Note that we also access the output in a coalesced manner across different blocks.

Compared to cuBLAS Gemm, we observe 8–20% performance speedup for running the vector-matrix multiplications used in a Transformer layer. For a larger batch (like 10), we see 5%–10% improvement over the cuBLAS (without using tensor cores). As we don’t get any benefit for larger batch sizes than 16, we enable tensor-core cuBLAS GeMMs and use the non-fused version of the kernels.

## Seamless pipeline from training to inference with automatic kernel-injection

To run the model in Inference mode, DeepSpeed simply requires the location of the model checkpoints and the desired parallelism configuration, i.e., MP/PP degree. DeepSpeed Inference kernels can also be enabled for many well-known model architectures such as HuggingFace (Bert and GPT-2) or Megatron GPT-based models using a pre-defined policy map that maps the original parameters to the parameters in the inference kernels. For other transformer-based models, user can specify their own policy map. Note that DS-Inference can run independent of the training pipeline as long as it receives all model checkpoints, and the DeepSpeed Transformer kernels for inference can be injected into any Transformer model if the right mapping policy is defined. For more information on how to enable Transformer inference kernel as well as specifying parallelism, please refer to out [inference tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/).


## Flexible quantization support

To further reduce the inference cost for large-scale models, we created the DeepSpeed Quantization Toolkit, supporting flexible quantize-aware training and high-performance kernels for quantized inference.

For training, we introduce a novel approach called Mixture of Quantization (MoQ), which is inspired by mixed-precision training while seamlessly applying quantization. With MoQ, we can control the precision of the model by simulating the impact of quantization when updating the parameters at each step of training. Moreover, it supports flexible quantization policies and schedules—we find that by dynamically adjusting the number of quantization bits during training, the final quantized model provides higher accuracy under the same compression ratio. To adapt to different tasks, MoQ can also leverage the second order information of models to detect their sensitivity to precision and adjust the quantization schedule and target accordingly.  

To maximize the performance gains from the quantization model, we provide inference kernels tailored for quantized models that reduce latency through optimizing data movement but do not require specialized hardware. Finally, our toolkit does not require any code changes on the client side, making it easy to use.

## Performance results

Boosting throughput and reducing inference cost.  Figure 3 shows the inference throughput per GPU for the three model sizes corresponding to the three Transformer networks, GPT-2, Turing-NLG, and GPT-3. DeepSpeed Inference increases in per-GPU throughput by 2 to 4 times when using the same precision of FP16 as the baseline.  By enabling quantization, we boost throughput further. We reach a throughput improvement of 3x for GPT-2, 5x for Turing-NLG, and 3x for a model that is similar in characteristics and size to GPT-3, which directly translates to 3–5x inference cost reduction on serving these large models. In addition, we achieve these throughput and cost improvements without compromising latency as shown in Figure 5.  

![Inference-Throughput](/assets/images/inference-throughput.png){: .align-center}

Figure 3: Inference throughput for different model sizes. DeepSpeed Inference achieves 3x to 5x higher throughput than baseline.

One source of inference cost reduction is through reducing the number of GPUs for hosting large models as shown in Figure 4.  The optimized GPU resources comes from 1) using inference-adapted parallelism, allowing users to adjust the model and pipeline parallelism degree from the trained model checkpoints, and 2) shrinking model memory footprint by half with INT8 quantization.  As shown in this figure, we use 2x less GPUs to run inference for the 17B model size by adapting the parallelism.  Together with INT8 quantization through DeepSpeed MoQ, we use 4x and 2x fewer GPUs for 17B and 175B sizes respectively.  

![Inference-Throughput](/assets/images/gpu-numbers.png){: .align-center}

Figure 4: Number of GPUs used for running inference on the different model sizes shown in Figure 4.

Reducing inference latency.  For the application scenarios where inference latency is critical, we can increase model parallelism degree in DeepSpeed Inference to reduce inference latency further.  As Figure 5 depicts, we can reduce the latency by 2.3x compared to PyTorch as we increase the model-parallelism size to 4.  Furthermore, we can still have high latency improvement with a fewer number of GPUs by adapting the parallelism at inference and using MoQ to quantize the model. We obtain 1.3x and 1.9x speedups while using 4x and 2x lower resources than baseline, respectively.

For the application scenarios where inference latency is critical, we can increase model parallelism degree in DeepSpeed Inference to reduce inference latency further.  As Figure 5 depicts, we can reduce the latency by 2.3x compared to PyTorch as we increase the model-parallelism size to 4.  Furthermore, we can still have high latency improvement with a fewer number of GPUs by adapting the parallelism at inference and using MoQ to quantize the model. We obtain 1.3x and 1.9x speedups while using 4x and 2x lower resources than baseline, respectively.

![Inference-Throughput](/assets/images/inference-latency.png){: .align-center}

Figure 5. Inference latency for the 17B model using different parallelism configuration to optimize latency.
