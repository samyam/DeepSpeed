from abc import ABC

import torch


class DSPolicy(ABC):
    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError

    def mlp(self):
        """
        Returns mlp intermediate and output
        weight: (intermediate, hidden) and (hidden, intermediate)
        bias: (intermediate) and (hidden)
        """
        raise NotImplementedError

    def layerNorm(self):
        """
        Returns LayerNorms used in transformer layer
        Post-Attention and pre/post layer norm
        gamma and beta with shape: (hidden)
        """
        raise NotImplementedError


class HFBertLayerPolicy(DSPolicy):
    def __init__(self, client_module, inference=False, preln=True):
        self.client_module = client_module
        self.inference = inference
        self.preln = preln

    def attention(self):
        if self.inference:
            self.client_module.attention.self.query.weight.data = \
                self.client_module.attention.self.query.weight.data.transpose(
                    -1,
                    -2).contiguous()
            self.client_module.attention.self.key.weight.data = \
                self.client_module.attention.self.key.weight.data.transpose(
                    -1,
                    -2).contiguous()
            self.client_module.attention.self.value.weight.data = \
                self.client_module.attention.self.value.weight.data.transpose(
                    -1,
                    -2).contiguous()
            self.client_module.attention.output.dense.weight.data = \
                    self.client_module.attention.output.dense.weight.data.transpose(
                    -1,
                    -2).contiguous()
        qw = self.client_module.attention.self.query.weight.data
        qb = self.client_module.attention.self.query.bias.data
        kw = self.client_module.attention.self.key.weight.data
        kb = self.client_module.attention.self.key.bias.data
        vw = self.client_module.attention.self.value.weight.data
        vb = self.client_module.attention.self.value.bias.data

        if self.inference:
            qkvw = torch.cat((qw, kw, vw), dim=1)
        else:
            qkvw = torch.cat((qw, kw, vw), dim=0)
        qkvb = torch.cat((qb, kb, vb), 0)

        return qkvw, qkvb, self.client_module.attention.output.dense.weight, \
            self.client_module.attention.output.dense.bias

    def mlp(self):
        if self.preln:
            intermediate_ff = self.client_module.intermediate.dense_act
        else:
            intermediate_ff = self.client_module.intermediate.dense
        if self.inference:
            intermediate_ff.weight.data = intermediate_ff.weight.data.transpose(
                -1,
                -2).contiguous()

            self.client_module.output.dense.weight.data = \
                self.client_module.output.dense.weight.data.transpose(
                -1,
                -2).contiguous()
        return intermediate_ff.weight, intermediate_ff.bias, \
            self.client_module.output.dense.weight, \
            self.client_module.output.dense.bias

    def layerNorm(self):
        if self.preln:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
            transformer_layernorm = self.client_module.PreAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
            transformer_layernorm = self.client_module.output.LayerNorm
        return attention_layernorm.weight, \
               attention_layernorm.bias, \
               transformer_layernorm.weight, \
               transformer_layernorm.bias


class MegatronLayerPolicy(DSPolicy):
    def __init__(self, client_module, inference=True):

        self.client_module = client_module
        self.inference = inference

    def attention(self):
        if self.inference:
            self.client_module.attention.query_key_value.weight.data = \
                self.client_module.attention.query_key_value.weight.data.transpose(
                -1,
                -2).contiguous()

            self.client_module.attention.dense.weight.data = \
                self.client_module.attention.dense.weight.data.transpose(
                -1,
                -2).contiguous()

        return self.client_module.attention.query_key_value.weight, \
                self.client_module.attention.query_key_value.bias, \
                self.client_module.attention.dense.weight, \
                self.client_module.attention.dense.bias

    def mlp(self):
        if self.inference:
            self.client_module.mlp.dense_h_to_4h.weight.data = \
                self.client_module.mlp.dense_h_to_4h.weight.data.transpose(
                -1,
                -2).contiguous()

            self.client_module.mlp.dense_4h_to_h.weight.data = \
                self.client_module.mlp.dense_4h_to_h.weight.data.transpose(
                -1,
                -2).contiguous()

        return self.client_module.mlp.dense_h_to_4h.weight, \
            self.client_module.mlp.dense_h_to_4h.bias, \
            self.client_module.mlp.dense_4h_to_h.weight, \
            self.client_module.mlp.dense_4h_to_h.bias

    def layerNorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias


class HFGPT2LayerPolicy(DSPolicy):
    def __init__(self, client_module, inference=True):
        self.client_module = client_module
        self.inference = inference

    def attention(self):
        return self.client_module.attn.c_attn.weight.data, \
                self.client_module.attn.c_attn.bias.data, \
                self.client_module.attn.c_proj.weight.data, \
                self.client_module.attn.c_proj.bias.data

    def mlp(self):
        return self.client_module.mlp.c_fc.weight.data, \
            self.client_module.mlp.c_fc.bias.data, \
            self.client_module.mlp.c_proj.weight.data, \
            self.client_module.mlp.c_proj.bias.data

    def layerNorm(self):
        return self.client_module.ln_2.weight.data, \
               self.client_module.ln_2.bias.data, \
               self.client_module.ln_1.weight.data, \
               self.client_module.ln_1.bias.data
