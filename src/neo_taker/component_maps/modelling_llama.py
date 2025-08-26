from transformers import AutoConfig
import einops
import torch
from typing import Optional, Any

from .map_data_classes import ModelMapData, MapConfigClass
from .map_utils import generate_sizes_dict, generate_attn_qkv_functions, update_param, get_attrs, set_attrs

def convert_hf_llama_config(hf_config: AutoConfig):

    cfg_dict = {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "n_key_value_heads": hf_config.num_key_value_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
        "final_rms": True,
        "gated_mlp": True,
    }

    return MapConfigClass(**cfg_dict)


# LLaMa Models
##############

llama_model_map = {
    "model"           : "model",
    "layers"          : "model.layers",
    "embed"           : "model.embed_tokens",
    "embed.W_E"       : "model.embed.weights",
    "pos_embed"       : "model.embed_positions",
    "pos_embed.W"     : "model.embed_positions.weight",
    "ln_final"        : "model.norm",
    "ln_final.w"      : "model.norm.weight",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
}

def build_llama_layer_map(cfg: MapConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.in_proj": "up_proj", "mlp.out_proj": "down_proj", "mlp.gate_proj": "gate_proj"}

    def llama_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def llama_attn_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Create fake bias with zeros because is easier to handle
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        attn = layer.self_attn
        _proj = get_attrs(attn, attn_proj_map[key]).weight
        b = torch.zeros(
            _proj.shape[:-1], dtype=_proj.dtype, device=_proj.device
        )
        if key == "o":
            return b
        return einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)


    def llama_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        mlp = layer.mlp
        _proj = get_attrs(mlp, mlp_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        return b

    def get_attn_weights(attn_outputs):
        attn_out, attn_weights, past_key_value = attn_outputs
        return attn_weights

    def set_attn_weights(attn_weights, orig_output):
        return orig_output[0], attn_weights, orig_output[2]


    llama_layer_map = {
        # === Attention Layers ===
        "attn.ln_in"    : "input_layernorm",
        "attn.ln_in.w"  : "input_layernorm.weight",
        "attn.ln_in.b"  : None,

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",
        
        **generate_attn_qkv_functions(llama_qkv_weight, llama_attn_bias),

        "attn.out_proj" : "self_attn.o_proj",
        "attn.W_O"      : "self_attn.o_proj.weight",
        "attn.b_O"      : lambda layer, _inpt=None: llama_attn_bias(layer, "o", _inpt),
        
        "attn.ln_out"   : None,
        "attn.ln_out.w" : None,
        "attn.ln_out.b" : None,

        # === MLP Layers ===
        "mlp.ln_in"     : "post_attention_layernorm",
        "mlp.ln_in.w"   : "post_attention_layernorm.weight",
        "mlp.ln_in.b"   : None,

        "mlp"           : "mlp",
        "mlp.in_proj"   : "mlp.up_proj",
        "mlp.gate_proj" : "mlp.gate_proj",
        "mlp.W_in"      : "mlp.up_proj.weight",
        "mlp.W_gate"    : "mlp.gate_proj.weight",
        "mlp.b_in"      : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.in_proj", _inpt),
        "mlp.b_gate"    : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.out_proj", _inpt),

        "activation_fn" : "mlp.act_fn",

        "mlp.out_proj"  : "mlp.down_proj",
        "mlp.W_out"     : "mlp.down_proj.weight",
        "mlp.b_out"     : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.gate_proj", _inpt),
        
        "mlp.ln_out"    : None,
        "mlp.ln_out.w"  : None,
        "mlp.ln_out.b"  : None,
        
        
        # === TransformerLens naming conventions ===
        # Layer norms (ln1 = pre-attention, ln2 = pre-MLP)
        "ln1"               : "input_layernorm",
        "ln1.weight"        : "input_layernorm.weight", 
        "ln1.bias"          : None,
        "ln2"               : "post_attention_layernorm",
        "ln2.weight"        : "post_attention_layernorm.weight",
        "ln2.bias"          : None,
        
        # Attention projections (TransformerLens style)
        "self_attn"         : "self_attn",
        "q_proj"            : "self_attn.q_proj",
        "k_proj"            : "self_attn.k_proj", 
        "v_proj"            : "self_attn.v_proj",
        "o_proj"            : "self_attn.o_proj",
        
        # MLP projections (TransformerLens style)
        "up_proj"           : "mlp.up_proj",
        "down_proj"         : "mlp.down_proj",
        "gate_proj"         : "mlp.gate_proj",
        
        # Weight matrices (TransformerLens W_ naming)
        "W_Q"               : lambda layer, _inpt=None: llama_qkv_weight(layer, "q", _inpt),
        "W_K"               : lambda layer, _inpt=None: llama_qkv_weight(layer, "k", _inpt),
        "W_V"               : lambda layer, _inpt=None: llama_qkv_weight(layer, "v", _inpt),
        "W_O"               : "self_attn.o_proj.weight",
        "W_in"              : "mlp.up_proj.weight",
        "W_out"             : "mlp.down_proj.weight", 
        "W_gate"            : "mlp.gate_proj.weight",
        
        # Bias vectors (TransformerLens b_ naming)  
        "b_Q"               : lambda layer, _inpt=None: llama_attn_bias(layer, "q", _inpt),
        "b_K"               : lambda layer, _inpt=None: llama_attn_bias(layer, "k", _inpt),
        "b_V"               : lambda layer, _inpt=None: llama_attn_bias(layer, "v", _inpt),
        "b_O"               : lambda layer, _inpt=None: llama_attn_bias(layer, "o", _inpt),
        "b_in"              : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.in_proj", _inpt),
        "b_out"             : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.gate_proj", _inpt),
    }
    return llama_layer_map

# Final Return
#####################################################################################

LlamaModelMapData = ModelMapData(
    config_map=convert_hf_llama_config,
    model_map_dict=llama_model_map,
    layer_map_factory=build_llama_layer_map,
)
