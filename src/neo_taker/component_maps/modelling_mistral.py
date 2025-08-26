from transformers import AutoConfig
from .data_classes import MapConfigClass

def convert_hf_mistral_config(hf_config: AutoConfig):

    elif architecture == "MistralForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "use_local_attn": True,
            "gated_mlp": True,
        }