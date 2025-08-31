from transformers import AutoConfig
import einops
from typing import Optional, Any

from .map_data_classes import ModelMapData, MapConfigClass
from .map_utils import generate_sizes_dict, generate_attn_qkv_functions, update_param, get_attrs


def convert_hf_gemma3_config(hf_config: AutoConfig):
    # Some fields may differ across HF releases; use getattr with sane fallbacks
    act_fn = getattr(hf_config, "hidden_activation", getattr(hf_config, "hidden_act", None))
    rope_theta = getattr(hf_config, "rope_theta", None)
    head_dim = getattr(hf_config, "head_dim", None)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", None))
    query_pre_attn_scalar = getattr(hf_config, "query_pre_attn_scalar", None)
    sliding_window = getattr(hf_config, "sliding_window", None)
    attn_logit_softcapping = getattr(hf_config, "attn_logit_softcapping", None)
    final_logit_softcapping = getattr(hf_config, "final_logit_softcapping", None)

    cfg_dict = {
        "d_model": hf_config.hidden_size,
        "d_head": head_dim if head_dim is not None else hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "n_key_value_heads": num_kv_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": act_fn,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_dim": head_dim if head_dim is not None else hf_config.hidden_size // hf_config.num_attention_heads,
        "rotary_base": rope_theta,
        "final_rms": True,
        "gated_mlp": True,
        "pre_layernorm": True,
        "post_layernorm": True,
        # Attention specifics
        "use_attn_scale": True if query_pre_attn_scalar is not None else None,
        "attn_scale": (query_pre_attn_scalar ** -0.5) if query_pre_attn_scalar is not None else None,
        "use_local_attn": sliding_window is not None,
        "window_size": sliding_window,
        "attn_types": (["global", "local"] * (hf_config.num_hidden_layers // 2)) if sliding_window else None,
        "attn_scores_soft_cap": attn_logit_softcapping,
        "output_logits_soft_cap": final_logit_softcapping,
    }
    return MapConfigClass(**cfg_dict)


# Gemma 3 (text-only)
######################

gemma3_model_map = {
    "model": "model",
    "layers": "model.layers",
    "embed": "model.embed_tokens",
    "embed.W_E": "model.embed_tokens.weight",
    # Positional embeddings handled by rotary in attention
    "ln_final": "model.norm",
    "ln_final.w": "model.norm.weight",
    "unembed": "lm_head",
    "unembed.W_U": "lm_head.weight.T",  # Linear, bias=False
    "unembed.b_U": None,
}


def build_gemma3_layer_map(cfg: MapConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.gate_proj": "gate_proj", "mlp.up_proj": "up_proj", "mlp.down_proj": "down_proj"}

    def gemma3_qkv_weight(layer, key: str, inpt: Optional[Any] = None):
        # Handle GQA: K/V use n_key_value_heads
        is_kv = key in ["k", "v"]
        num_heads = cfg.n_key_value_heads if is_kv and cfg.n_key_value_heads is not None else cfg.n_heads
        their_shape = f"({num_heads} d_head) d_model"
        my_shape = f"{num_heads} d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)
        sizes["n_heads"] = num_heads  # Override for GQA case

        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def gemma3_attn_bias(layer, key: str, _inpt: Optional[Any] = None):
        # Gemma3 defaults to bias=False on attention projections
        return None

    def gemma3_mlp_weight(layer, key: str, inpt: Optional[Any] = None):
        mlp = layer.mlp
        proj = get_attrs(mlp, mlp_proj_map[key])
        if inpt is None:
            return proj.weight
        update_param(proj, "weight", inpt)

    def gemma3_mlp_bias(layer, key: str, _inpt: Optional[Any] = None):
        # Gemma3 MLP projections use bias=False
        return None

    gemma3_layer_map = {
        # === Attention Norm (pre-attn) ===
        "attn.ln_in": "input_layernorm",
        "attn.ln_in.w": "input_layernorm.weight",
        "attn.ln_in.b": None,  # RMSNorm

        # === Attention Projections ===
        "attn": "self_attn",
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        **generate_attn_qkv_functions(gemma3_qkv_weight, gemma3_attn_bias),

        # === Attention Output ===
        "attn.out_proj": "self_attn.o_proj",
        "attn.W_O": "self_attn.o_proj.weight",
        "attn.b_O": lambda layer, _inpt=None: gemma3_attn_bias(layer, "o", _inpt),

        # === Post-Attention Norm ===
        "attn.ln_out": "post_attention_layernorm",
        "attn.ln_out.w": "post_attention_layernorm.weight",
        "attn.ln_out.b": None,  # RMSNorm

        # === Pre-MLP Norm ===
        "mlp.ln_in": "pre_feedforward_layernorm",
        "mlp.ln_in.w": "pre_feedforward_layernorm.weight",
        "mlp.ln_in.b": None,  # RMSNorm

        # === MLP ===
        "mlp": "mlp",
        "mlp.gate_proj": "mlp.gate_proj",
        "mlp.up_proj": "mlp.up_proj",
        "mlp.out_proj": "mlp.down_proj",
        "mlp.W_gate": lambda layer, inpt=None: gemma3_mlp_weight(layer, "mlp.gate_proj", inpt),
        "mlp.W_in": lambda layer, inpt=None: gemma3_mlp_weight(layer, "mlp.up_proj", inpt),
        "mlp.W_out": lambda layer, inpt=None: gemma3_mlp_weight(layer, "mlp.down_proj", inpt),
        "mlp.b_gate": lambda layer, _inpt=None: gemma3_mlp_bias(layer, "mlp.gate_proj", _inpt),
        "mlp.b_in": lambda layer, _inpt=None: gemma3_mlp_bias(layer, "mlp.up_proj", _inpt),
        "mlp.b_out": lambda layer, _inpt=None: gemma3_mlp_bias(layer, "mlp.down_proj", _inpt),

        # === Activation ===
        "activation_fn": "mlp.act_fn",

        # === Post-MLP Norm ===
        "mlp.ln_out": "post_feedforward_layernorm",
        "mlp.ln_out.w": "post_feedforward_layernorm.weight",
        "mlp.ln_out.b": None,  # RMSNorm

        # === TransformerLens-style aliases ===
        # Layer norms
        "ln1": "input_layernorm",
        "ln1.weight": "input_layernorm.weight",
        "ln1.bias": None,
        "ln2": "post_attention_layernorm",
        "ln2.weight": "post_attention_layernorm.weight",
        "ln2.bias": None,

        # Attention modules (TL names)
        "self_attn": "self_attn",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",

        # MLP projections (TL names)
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
        "gate_proj": "mlp.gate_proj",

        # Weight matrices (TL W_ naming)
        "W_Q": lambda layer, _inpt=None: gemma3_qkv_weight(layer, "q", _inpt),
        "W_K": lambda layer, _inpt=None: gemma3_qkv_weight(layer, "k", _inpt),
        "W_V": lambda layer, _inpt=None: gemma3_qkv_weight(layer, "v", _inpt),
        "W_O": "self_attn.o_proj.weight",
        "W_in": "mlp.up_proj.weight",
        "W_out": "mlp.down_proj.weight",
        "W_gate": "mlp.gate_proj.weight",

        # Bias vectors (TL b_ naming)
        "b_Q": lambda layer, _inpt=None: gemma3_attn_bias(layer, "q", _inpt),
        "b_K": lambda layer, _inpt=None: gemma3_attn_bias(layer, "k", _inpt),
        "b_V": lambda layer, _inpt=None: gemma3_attn_bias(layer, "v", _inpt),
        "b_O": lambda layer, _inpt=None: gemma3_attn_bias(layer, "o", _inpt),
        "b_in": lambda layer, _inpt=None: gemma3_mlp_bias(layer, "mlp.up_proj", _inpt),
        "b_out": lambda layer, _inpt=None: gemma3_mlp_bias(layer, "mlp.down_proj", _inpt),
    }

    return gemma3_layer_map


# Final Return
#####################################################################################

Gemma3ModelMapData = ModelMapData(
    config_map=convert_hf_gemma3_config,
    model_map_dict=gemma3_model_map,
    layer_map_factory=build_gemma3_layer_map,
)