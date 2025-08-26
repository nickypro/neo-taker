from transformers import AutoConfig
import einops
import torch
from typing import Optional, Any

from .map_data_classes import ModelMapData, MapConfigClass
from .map_utils import generate_sizes_dict, generate_attn_qkv_functions, update_param, get_attrs, set_attrs

def convert_hf_gpt2_config(hf_config: AutoConfig):

    cfg_dict = {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.n_embd * 4,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_ctx,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
        "pre_layernorm": False,
        "post_layernorm": True,
        "positional_embedding_type": "standard",
        "final_rms": False,
        "gated_mlp": False,
    }

    return MapConfigClass(**cfg_dict)


# GPT2 Models
#############

gpt2_model_map = {
    "model"           : "transformer",
    "layers"          : "transformer.h",
    "embed"           : "transformer.wte",
    "embed.W_E"       : "transformer.wte.weight",
    "pos_embed.W_pos" : "transformer.wpe.weight",
    "ln_final"        : "transformer.ln_f",
    "ln_final.w"      : "transformer.ln_f.weight",
    "ln_final.b"      : "transformer.ln_f.bias",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
    "unembed.b_U"     : None,
}

def build_gpt2_layer_map(cfg: MapConfigClass):
    def gpt2_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        their_shape = "d_model (qkv n_heads d_head)"
        my_shape    = "qkv n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head weights
        qkv_heads = layer.attn.c_attn
        W = qkv_heads.weight
        W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return W[index]

        # Set mode
        W[index] = inpt
        W = einops.rearrange(W, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_heads, "weight", W)

    def gpt2_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        their_shape = "(qkv n_heads d_head)"
        my_shape    = "qkv n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head biases
        qkv_heads = layer.attn.c_attn
        qkv_bias = qkv_heads.bias
        qkv_bias = einops.rearrange(qkv_bias, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return qkv_bias[index]

        # Set mode
        qkv_bias[index] = inpt
        qkv_bias = einops.rearrange(qkv_bias, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_heads, "bias", qkv_bias)

    # GPT2 uses Conv1D instead of Linear, so we must get the transpose
    def conv1d_weight(module, inpt=None):
        if inpt is None:
            return module.weight.T
        params = module.state_dict()
        params["weight"] = inpt.T
        module.load_state_dict(params)

    def gpt2_out_weight(layer, inpt=None):
        return conv1d_weight(layer.attn.c_proj, inpt)

    def gpt2_mlp_in_weight(layer, inpt=None):
        return conv1d_weight(layer.mlp.c_fc, inpt)

    def gpt2_mlp_out_weight(layer, inpt=None):
        return conv1d_weight(layer.mlp.c_proj, inpt)

    def get_attn_weights(attn_outputs):
        # outputs # a, present, (attentions)
        # TODO: make sure use_cache is true!
        attn_out, key_value_cache, attn_weights = attn_outputs
        return attn_weights

    def set_attn_weights(attn_weights, orig_output):
        return orig_output[0], attn_weights, orig_output[2]


    gpt2_layer_map = {
        # === Attention Layers ===
        "attn.ln_in"    : None,  # GPT-2 has post-layer norm
        "attn.ln_in.w"  : None,
        "attn.ln_in.b"  : None,

        "attn"          : "attn",
        # GPT-2 combines QKV in a single Conv1D `c_attn`. We map q/k/v proj
        # to the same module so hooks in model.py can attach and split.
        "attn.q_proj"   : "attn.c_attn",
        "attn.k_proj"   : "attn.c_attn",
        "attn.v_proj"   : "attn.c_attn",
        
        **generate_attn_qkv_functions(gpt2_qkv_weight, gpt2_qkv_bias),

        "attn.out_proj" : "attn.c_proj",
        "attn.W_O"      : lambda layer, inpt=None: gpt2_out_weight(layer, inpt),
        "attn.b_O"      : "attn.c_proj.bias",
        
        "attn.ln_out"   : "ln_1",  # GPT-2 has post-layer norm
        "attn.ln_out.w" : "ln_1.weight",
        "attn.ln_out.b" : "ln_1.bias",

        # === MLP Layers ===
        "mlp.ln_in"     : None,  # GPT-2 has post-layer norm
        "mlp.ln_in.w"   : None,
        "mlp.ln_in.b"   : None,

        "mlp"           : "mlp",
        "mlp.in_proj"   : "mlp.c_fc",
        "mlp.gate_proj" : None,  # GPT-2 doesn't have gated MLP
        "mlp.W_in"      : lambda layer, inpt=None: gpt2_mlp_in_weight(layer, inpt),
        "mlp.W_gate"    : None,  # GPT-2 doesn't have gated MLP
        "mlp.b_in"      : "mlp.c_fc.bias",
        "mlp.b_gate"    : None,  # GPT-2 doesn't have gated MLP

        "activation_fn" : "mlp.act",

        "mlp.out_proj"  : "mlp.c_proj",
        "mlp.W_out"     : lambda layer, inpt=None: gpt2_mlp_out_weight(layer, inpt),
        "mlp.b_out"     : "mlp.c_proj.bias",
        
        "mlp.ln_out"    : "ln_2",  # GPT-2 has post-layer norm
        "mlp.ln_out.w"  : "ln_2.weight",
        "mlp.ln_out.b"  : "ln_2.bias",
        
        
        # === TransformerLens naming conventions ===
        # Layer norms (ln1 = post-attention, ln2 = post-MLP in GPT-2)
        "ln1"               : "ln_1",
        "ln1.weight"        : "ln_1.weight", 
        "ln1.bias"          : "ln_1.bias",
        "ln2"               : "ln_2",
        "ln2.weight"        : "ln_2.weight",
        "ln2.bias"          : "ln_2.bias",
        
        # Attention projections (TransformerLens style)
        "self_attn"         : "attn",
        # Expose TL-style names for hooking. All point to c_attn.
        "q_proj"            : "attn.c_attn",
        "k_proj"            : "attn.c_attn",
        "v_proj"            : "attn.c_attn",
        "o_proj"            : "attn.c_proj",
        
        # MLP projections (TransformerLens style)
        "in_proj"           : "mlp.c_fc",
        "out_proj"          : "mlp.c_proj",
        
        # Weight matrices (TransformerLens W_ naming)
        "W_Q"               : lambda layer, _inpt=None: gpt2_qkv_weight(layer, "q", _inpt),
        "W_K"               : lambda layer, _inpt=None: gpt2_qkv_weight(layer, "k", _inpt),
        "W_V"               : lambda layer, _inpt=None: gpt2_qkv_weight(layer, "v", _inpt),
        "W_O"               : lambda layer, inpt=None: gpt2_out_weight(layer, inpt),
        "W_in"              : lambda layer, inpt=None: gpt2_mlp_in_weight(layer, inpt),
        "W_out"             : lambda layer, inpt=None: gpt2_mlp_out_weight(layer, inpt),
        
        # Bias vectors (TransformerLens b_ naming)  
        "b_Q"               : lambda layer, _inpt=None: gpt2_qkv_bias(layer, "q", _inpt),
        "b_K"               : lambda layer, _inpt=None: gpt2_qkv_bias(layer, "k", _inpt),
        "b_V"               : lambda layer, _inpt=None: gpt2_qkv_bias(layer, "v", _inpt),
        "b_O"               : "attn.c_proj.bias",
        "b_in"              : "mlp.c_fc.bias",
        "b_out"             : "mlp.c_proj.bias",
    }
    return gpt2_layer_map

# Final Return
#####################################################################################

GPT2ModelMapData = ModelMapData(
    config_map=convert_hf_gpt2_config,
    model_map_dict=gpt2_model_map,
    layer_map_factory=build_gpt2_layer_map,
)
