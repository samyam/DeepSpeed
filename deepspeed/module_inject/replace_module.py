import copy
import torch
import deepspeed
import deepspeed.ops.transformer as transformer_inference
from .replace_policy import HFBertLayerPolicy, MegatronLayerPolicy


def replace_transformer_layer(orig_layer_impl,
                              model,
                              policy=HFBertLayerPolicy,
                              micro_batch_size=-1,
                              bert_config=None,
                              seed=-1,
                              hidden_size=-1,
                              num_attention_heads=-1,
                              mp_size=1,
                              mp_group=None,
                              preln=True,
                              fp16=True,
                              local_rank=-1,
                              training=True,
                              quantize=False,
                              encoder_decoder=False,
                              quantize_settings=None):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        policy: shows the policy for mapping from the orig_layer_impl to transformer parameters
        micro_batch_size (int): micro batch size per gpu used during training/eval
        bert_config (dict): model config containing hidden size, attention heads, etc.
        seed (int): random seed value
        max_seq_length (int): max sequence length for training
        hidden_size (int): hidden dimension
        num_attention_heads (int): numebr of attention heads
        mp_size (int): model_parallelism degree
        mp_group : model_parallel gropu initialized on the modeling side
        preln (bool): does the original layer implementation do pre or post layer norm?
        fp16 (bool): fp16 or fp32
        local_rank (int): GPU rank (optional),
        training (bool): specifying whether kernel-injection is done for training/inference (set to false for inference-mode injection)

        Note: For Bert kind of models, we inject based on the DeepSpeed-Example models, if not setting huggingface flag.

    Returns:
        Updated nn.module with replaced transformer layers
    """
    def replace_with_policy(new_module, child, policy_cls, inference=False, preln=True):
        if policy_cls is HFBertLayerPolicy:
            policy = policy_cls(child, inference=inference, preln=preln)
        else:
            policy = policy_cls(child, inference=inference)

        qkvw, qkvb, dense_w, dense_b = policy.attention()
        _h4h_w, _h4h_b, _4hh_w, _4hh_b = policy.mlp()
        attn_nw, attn_nb, input_nw, input_nb = policy.layerNorm()

        if inference:
            new_module.attention.attn_qkvw.data = qkvw
            new_module.attention.attn_qkvb.data = qkvb
            new_module.attention.attn_ow.data = dense_w
            new_module.attention.attn_ob.data = dense_b

            new_module.mlp.inter_w.data = _h4h_w
            new_module.mlp.inter_b.data = _h4h_b
            new_module.mlp.output_w.data = _4hh_w
            new_module.mlp.output_b.data = _4hh_b

            new_module.mlp.attn_nw.data = attn_nw
            new_module.mlp.attn_nb.data = attn_nb
            new_module.norm_w.data = input_nw
            new_module.norm_b.data = input_nb
        else:
            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = dense_w
            new_module.attn_ob.data = dense_b

            new_module.attn_nw.data = attn_nw
            new_module.attn_nb.data = attn_nb
            new_module.norm_w.data = input_nw
            new_module.norm_b.data = input_nb

            new_module.inter_w.data = _h4h_w
            new_module.inter_b.data = _h4h_b
            new_module.output_w.data = _4hh_w
            new_module.output_b.data = _4hh_b

    def replace_fn(child, layer_id=0):
        if training:
            transformer_config = deepspeed.DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                hidden_size=bert_config.hidden_size,
                heads=bert_config.num_attention_heads,
                attn_dropout_ratio=bert_config.attention_probs_dropout_prob,
                hidden_dropout_ratio=bert_config.hidden_dropout_prob,
                num_hidden_layers=bert_config.num_hidden_layers,
                initializer_range=bert_config.initializer_range,
                seed=seed,
                fp16=fp16,
                pre_layer_norm=preln,
                huggingface=encoder_decoder,
                local_rank=local_rank,
                training=training)
            new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)

            # copy relevant state from child -> new module
            replace_with_policy(new_module, child, policy, preln=preln)

        else:
            transformer_config = transformer_inference.DeepSpeedInferenceConfig(
                hidden_size=hidden_size,
                heads=num_attention_heads,
                fp16=fp16,
                pre_layer_norm=preln,
                mp_size=mp_size,
                q_int8=quantize,
                encoder_decoder=encoder_decoder)

            if quantize and quantize_settings is not None:
                (quantization_scales,
                 merge_count,
                 mlp_extra_grouping,
                 quantize_groups) = quantize_settings
                new_module = transformer_inference.DeepSpeedTransformerInference(
                    transformer_config,
                    mp_group=mp_group,
                    quantize_scales=quantization_scales[layer_id],
                    quantize_groups=quantize_groups,
                    merge_count=merge_count,
                    mlp_extra_grouping=mlp_extra_grouping)
            else:
                new_module = transformer_inference.DeepSpeedTransformerInference(
                    transformer_config,
                    mp_group=mp_group,
                )

            # copy relevant state from child -> new module
            replace_with_policy(new_module, child, policy, inference=True, preln=preln)

        return new_module

    return replace_module(model=model, orig_class=orig_layer_impl, replace_fn=replace_fn)


def revert_transformer_layer(orig_layer_impl, model, bert_config, preln=False):
    """ Revert DeepSpeed's transformer layer back to original bert-style transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation that was replaced,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        bert_config (dict): model config containing hidden size, attention heads, etc.

    Returns:
        Updated nn.module with original bert-style transformer layers
    """
    def replace_fn(child):
        #from turing.nvidia_modelingpreln import BertLayer
        orig_module = orig_layer_impl(bert_config)

        # copy relevant state from child -> original module
        qkvw = child.attn_qkvw.data
        qkvb = child.attn_qkvb.data

        qw, kw, vw = torch.chunk(qkvw, 3, axis=0)
        qb, kb, vb = torch.chunk(qkvb, 3, axis=0)

        orig_module.attention.self.query.weight.data = qw
        orig_module.attention.self.query.bias.data = qb
        orig_module.attention.self.key.weight.data = kw
        orig_module.attention.self.key.bias.data = kb
        orig_module.attention.self.value.weight.data = vw
        orig_module.attention.self.value.bias.data = vb

        orig_module.attention.output.dense.weight.data = child.attn_ow.data
        orig_module.attention.output.dense.bias.data = child.attn_ob.data

        attn_ln_w = child.attn_nw.data
        attn_ln_b = child.attn_nb.data
        if preln:
            orig_module.PostAttentionLayerNorm.weight.data = attn_ln_w
            orig_module.PostAttentionLayerNorm.bias.data = attn_ln_b
        else:
            orig_module.attention.output.LayerNorm.weight.data = attn_ln_w
            orig_module.attention.output.LayerNorm.bias.data = attn_ln_b

        inter_ff_w = child.inter_w.data
        inter_ff_b = child.inter_b.data
        if preln:
            orig_module.intermediate.dense_act.weight.data = inter_ff_w
            orig_module.intermediate.dense_act.bias.data = inter_ff_b
        else:
            orig_module.intermediate.dense.weight.data = inter_ff_w
            orig_module.intermediate.dense.bias.data = inter_ff_b

        orig_module.output.dense.weight.data = child.output_w.data
        orig_module.output.dense.bias.data = child.output_b.data

        transformer_ln_w = child.norm_w.data
        transformer_ln_b = child.norm_b.data
        if preln:
            orig_module.PreAttentionLayerNorm.weight.data = transformer_ln_w
            orig_module.PreAttentionLayerNorm.bias.data = transformer_ln_b
        else:
            orig_module.output.LayerNorm.weight.data = transformer_ln_w
            orig_module.output.LayerNorm.bias.data = transformer_ln_b
        return orig_module

    return replace_module(model=model,
                          orig_class=deepspeed.DeepSpeedTransformerLayer,
                          replace_fn=replace_fn)


def replace_module(model, orig_class, replace_fn):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.

    Returns:
        A modified ``model``.
    """
    policy = {orig_class: replace_fn}
    replaced_module, _ = _replace_module(model, policy)
    return replaced_module


def _replace_module(model, policies, layer_id=0):
    """ Traverse model's children recursively and apply any transformations in ``policies``.
    Arguments:
        model (torch.nn.Module): model to augment
        policies (dict): Mapping of source class to replacement function.

    Returns:
        Modified ``model``.
    """
    for name, child in model.named_children():
        if child.__class__ in policies:
            orig = repr(child)
            setattr(model, name, policies[child.__class__](child, layer_id))
            new = getattr(model, name)
            layer_id += 1
        else:
            _, layer_id = _replace_module(child, policies, layer_id=layer_id)

    return model, layer_id
