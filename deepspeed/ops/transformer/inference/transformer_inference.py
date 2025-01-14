'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import json
import math
import importlib
import torch
from torch import nn
from torch.autograd import Function
import time
from ...op_builder import InferenceBuilder
# Cuda modules will be imported if needed
inference_cuda_module = None
import torch.nn as nn


class TransformerConfig():
    def __init__(self, hidden_size, intermediate_size, heads, num_hidden_layers):
        self.layer_id = -1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.heads = heads
        self.num_hidden_layers = num_hidden_layers


class DeepSpeedInferenceConfig(TransformerConfig):
    """Initialize the DeepSpeed Transformer Config.

        Arguments:
            hidden_size: The hidden size of the transformer layer

            intermediate_size: The intermediate size of the feed-forward part of transformer layer

            heads: The number of heads in the self-attention of the transformer layer

            num_hidden_layers: The number of transformer layers

            local_rank: Optional: The rank of GPU running the transformer kernel, it is not required
                to use if the model already set the current device, otherwise need to set it
                so that the transformer kernel can work on the right device

            mp_size (optional): This argument is mainly used to create the parameters on the kernel side
                using model-parallel architecture. If the client model already takes care of this, there is no
                need to pass this argument.

            fp16: Enable half-precision computation

            pre_layer_norm: Select between Pre-LN or Post-LN transformer architecture

            stochastic_mode:  Enable for high performance, please note that this flag has some level of
                non-determinism and can produce different results on different runs.  However, we have seen
                that by enabling it, the pretraining tasks such as BERT are not affected and can obtain
                a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend
                to turn it off in order to be able to reproduce the same result through the regular kernel execution.

            encoder_decoder: DeepSpeed-Inference currently support the encoder-only architecture! We will add
                the required features to support both soon!


    """
    def __init__(self,
                 hidden_size=-1,
                 intermediate_size=-1,
                 heads=-1,
                 num_hidden_layers=-1,
                 local_rank=-1,
                 mp_size=1,
                 fp16=False,
                 q_int8=False,
                 pre_layer_norm=True,
                 stochastic_mode=False,
                 encoder_decoder=False):
        super(DeepSpeedInferenceConfig,
              self).__init__(
                  hidden_size,
                  (intermediate_size if intermediate_size > 0 else 4 * hidden_size),
                  heads,
                  num_hidden_layers)
        self.fp16 = fp16
        self.pre_layer_norm = pre_layer_norm
        self.local_rank = local_rank
        self.stochastic_mode = stochastic_mode
        self.epsilon = 1.0e-5  # 1e-12
        self.mp_size = mp_size
        self.q_int8 = q_int8
        self.encoder_decoder = encoder_decoder

    @classmethod
    def from_dict(cls, json_object):
        config = DeepSpeedInferenceConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


class DeepSpeedSelfAttentionFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                input_mask,
                head_mask,
                layer_past,
                get_present,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                norm_w,
                norm_b,
                config,
                attn_qkvw,
                attn_qkvb,
                num_attention_heads_per_partition,
                norm_factor,
                hidden_size_per_partition,
                attn_ow,
                attn_ob,
                mp_group,
                q_scales,
                q_groups,
                merge_count):
        def _transpose_for_scores(x, key=False, reshape=False):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition,
                                           attention_head_size)
            x_1 = x.view(*new_x_shape)
            if key:
                x_1 = x_1.permute(0, 2, 3, 1)
            else:
                x_1 = x_1.permute(0, 2, 1, 3)
            if reshape:
                return x_1.reshape(x.shape)
            return x_1

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_layer_shape = x.size()[:-2] + \
                                      (hidden_size_per_partition,)
            return x.view(*new_x_layer_shape)

        def backup_attention(mixed_query, key_layer, value_layer):
            if layer_past is not None:
                past_key, past_value = layer_past
                key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=-2)
                value_layer = torch.cat((past_value.type_as(value_layer),
                                         value_layer),
                                        dim=-2)
            query = _transpose_for_scores(mixed_query, False)
            key = _transpose_for_scores(key_layer, True)
            value = _transpose_for_scores(value_layer, False)
            p = torch.matmul(query, key)
            ds_softmax = inference_cuda_module.softmax_fp16 if config.fp16 else \
                            inference_cuda_module.softmax_fp32
            p = ds_softmax(p / (float(key.size(-2))**0.5), input_mask, 0)
            context_layer = torch.matmul(p, value_layer)
            return context_layer, key_layer, value_layer

        def compute_attention(qkv_out):
            score_context_func = inference_cuda_module.softmax_context_fp16 if config.fp16 else \
                                    inference_cuda_module.softmax_context_fp32
            (mixed_query,
             key_layer,
             value_layer) = torch.split(qkv_out,
                                        (qkv_out.shape[-1] // 3),
                                        dim=(qkv_out.dim() - 1))
            if layer_past is not None:
                past_key, past_value = layer_past
                (context_layer,
                 key_layer,
                 value_layer) = score_context_func(mixed_query,
                                                   past_key.type_as(key_layer),
                                                   key_layer,
                                                   input_mask,
                                                   past_value.type_as(value_layer),
                                                   value_layer,
                                                   num_attention_heads_per_partition,
                                                   1 / norm_factor,
                                                   True)
            else:
                if mixed_query.shape[1] >= 32:
                    if layer_past is not None:
                        past_key, past_value = layer_past
                        key_layer = torch.cat((past_key.type_as(key_layer),
                                               key_layer),
                                              dim=-2)
                        value_layer = torch.cat((past_value.type_as(value_layer),
                                                 value_layer),
                                                dim=-2)
                    query = _transpose_for_scores(mixed_query, False, True)
                    key_layer1 = _transpose_for_scores(key_layer, True, True)
                    value_layer1 = _transpose_for_scores(value_layer, False, True)

                    context_layer, _, _ = score_context_func(query,
                                                            key_layer1 / norm_factor,
                                                            torch.empty(1),
                                                            input_mask,
                                                            value_layer1,
                                                            torch.empty(1),
                                                            num_attention_heads_per_partition,
                                                            norm_factor,
                                                            True)

                else:
                    (context_layer,
                     key_layer,
                     value_layer) = score_context_func(
                         mixed_query,
                         key_layer,
                         torch.empty(1),
                         input_mask,
                         value_layer,
                         torch.empty(1),
                         num_attention_heads_per_partition,
                         1 / norm_factor,
                         True)
            # Transpose Context
            context_layer = _transpose_for_context(context_layer)
            return context_layer, key_layer, value_layer

        def selfAttention_fp():
            vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                    inference_cuda_module.vector_matmul_fp32
            if not config.pre_layer_norm:
                linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else \
                                    inference_cuda_module.linear_layer_fp32

                qkv_out = linear_func(input, attn_qkvw, attn_qkvb)
            else:
                qkv_func = inference_cuda_module.qkv_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.qkv_gemm_fp32
                qkv_out = qkv_func(input,
                                   attn_qkvw,
                                   attn_qkvb,
                                   norm_w,
                                   norm_b,
                                   config.epsilon)
            context_layer, key_layer, value_layer = compute_attention(qkv_out)
            output = vector_matmul_func(context_layer, attn_ow)

            return output, key_layer, value_layer, context_layer

        def selfAttention_int8():
            if not config.pre_layer_norm:
                qkv_out = inference_cuda_module.linear_layer_int8(
                    input,
                    attn_qkvw,
                    attn_qkvb,
                    q_scales[0],
                    (q_groups * (2**merge_count)))
            else:
                qkv_out = inference_cuda_module.qkv_gemm_int8(input,
                                                              attn_qkvw,
                                                              attn_qkvb,
                                                              norm_w,
                                                              norm_b,
                                                              config.epsilon,
                                                              q_scales[0],
                                                              (q_groups *
                                                               (2**merge_count)))
            context_layer, key_layer, value_layer = compute_attention(qkv_out)
            output = inference_cuda_module.vector_matmul_int8(context_layer,
                                                              attn_ow,
                                                              q_scales[1],
                                                              q_groups,
                                                              (merge_count))
            return output, key_layer, value_layer, context_layer

        if config.q_int8:
            output, key_layer, value_layer, context_layer = selfAttention_int8()
        else:
            output, key_layer, value_layer, context_layer = selfAttention_fp()

        if mp_group is not None and torch.distributed.get_world_size(group=mp_group) > 1:
            torch.distributed.all_reduce(output, group=mp_group)

        return (output, key_layer, value_layer, context_layer)

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedSelfAttention(nn.Module):
    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config

        self.attn_qkvw = nn.Parameter(
            torch.Tensor((self.config.hidden_size // self.config.mp_size) * 3,
                         self.config.hidden_size))
        self.attn_qkvb = nn.Parameter(
            torch.Tensor((self.config.hidden_size // self.config.mp_size) * 3))

        self.attn_ow = nn.Parameter(
            torch.Tensor(self.config.hidden_size,
                         self.config.hidden_size // self.config.mp_size))

        self.attn_ob = nn.Parameter(torch.Tensor(self.config.hidden_size))

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(
            math.sqrt(self.config.hidden_size // self.config.heads))

    def forward(self,
                input,
                input_mask,
                head_mask=None,
                layer_past=None,
                get_present=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                norm_w=None,
                norm_b=None):
        output = DeepSpeedSelfAttentionFunction.apply(
            input,
            input_mask,
            head_mask,
            layer_past,
            get_present,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            norm_w,
            norm_b,
            self.config,
            self.attn_qkvw,
            self.attn_qkvb,
            self.num_attention_heads_per_partition,
            self.norm_factor,
            self.hidden_size_per_partition,
            self.attn_ow,
            self.attn_ob,
            self.mp_group,
            self.q_scales,
            self.q_groups,
            self.merge_count)

        return output


class DeepSpeedMLPFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                residual,
                bias,
                inter_w,
                inter_b,
                attn_nw,
                attn_nb,
                config,
                mp_group,
                output_b,
                output_w,
                q_scales,
                q_groups,
                merge_count):

        if config.q_int8:
            (intermediate,
             residual_add) = inference_cuda_module.mlp_gemm_int8(
                 input,
                 residual,
                 bias,
                 inter_w,
                 inter_b,
                 attn_nw,
                 attn_nb,
                 config.epsilon,
                 q_scales[2],
                 (q_groups * (2**merge_count)))
            output = inference_cuda_module.vector_matmul_int8(intermediate,
                                                              output_w,
                                                              q_scales[3],
                                                              q_groups,
                                                              (merge_count))
        else:
            mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.mlp_gemm_fp32
            vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                    inference_cuda_module.vector_matmul_fp32
            (intermediate,
             residual_add) = mlp_gemm_func(input,
                                           residual,
                                           bias,
                                           inter_w,
                                           inter_b,
                                           attn_nw,
                                           attn_nb,
                                           config.epsilon)
            output = vector_matmul_func(intermediate, output_w)

        if mp_group is not None and torch.distributed.get_world_size(group=mp_group) > 1:
            torch.distributed.all_reduce(output, group=mp_group)

        bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 else \
                                    inference_cuda_module.bias_residual_fp32

        output = bias_residual_func(output, residual_add, output_b)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedMLP(nn.Module):
    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=True):
        super(DeepSpeedMLP, self).__init__()

        self.config = config
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.inter_w = nn.Parameter(
            torch.Tensor(self.config.intermediate_size // self.config.mp_size,
                         self.config.hidden_size))
        self.inter_b = nn.Parameter(
            torch.Tensor(self.config.intermediate_size // self.config.mp_size))
        self.output_w = nn.Parameter(
            torch.Tensor(self.config.hidden_size,
                         (self.config.intermediate_size // self.config.mp_size)))
        self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))

        self.mp_group = mp_group

    def forward(self, input, residual, bias):
        return DeepSpeedMLPFunction.apply(input,
                                          residual,
                                          bias,
                                          self.inter_w,
                                          self.inter_b,
                                          self.attn_nw,
                                          self.attn_nb,
                                          self.config,
                                          self.mp_group,
                                          self.output_b,
                                          self.output_w,
                                          self.q_scales,
                                          self.q_groups,
                                          self.merge_count)


class DeepSpeedTransformerInference(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.

        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated

            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This arguement groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=True):
        super(DeepSpeedTransformerInference, self).__init__()

        self.config = config
        assert (self.config.encoder_decoder is False), \
            "DeepSpeed-Inference currently does not support encoder-decoder mode!"

        self.attention = DeepSpeedSelfAttention(config,
                                                mp_group,
                                                quantize_scales,
                                                quantize_groups,
                                                merge_count)
        self.mlp = DeepSpeedMLP(config,
                                mp_group,
                                quantize_scales,
                                quantize_groups,
                                merge_count,
                                mlp_extra_grouping)

        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))

        print("DeepSpeed Transformer Inference config is ", self.config.__dict__)

        global inference_cuda_module
        if inference_cuda_module is None:
            inference_cuda_module = InferenceBuilder().load()

    def forward(self,
                input,
                input_mask=None,
                attention_mask=None,
                head_mask=None,
                layer_past=None,
                get_key_value=False,
                get_present=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False):

        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask
        attention_output = self.attention(input,
                                          input_mask,
                                          head_mask,
                                          layer_past,
                                          get_present,
                                          encoder_hidden_states,
                                          encoder_attention_mask,
                                          output_attentions,
                                          self.norm_w,
                                          self.norm_b)

        if get_present:
            attention_output, p_key, p_value, _ = attention_output
            presents = (p_key, p_value)
        elif output_attentions:
            attention_output, _, _, context_output = attention_output
        else:
            attention_output, _, _, _ = attention_output
        output = self.mlp(attention_output, input, self.attention.attn_ob)
        if get_present:
            output = (output, presents)

        if not self.config.pre_layer_norm:
            if not self.config.q_int8:
                ds_layernorm = inference_cuda_module.layer_norm_fp16 if self.config.fp16 else \
                                        inference_cuda_module.layer_norm_fp32
                output = ds_layernorm(output,
                                      self.norm_w,
                                      self.norm_b,
                                      self.config.epsilon)
        return output
