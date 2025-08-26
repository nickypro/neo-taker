from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any
import torch

@dataclass
class MapConfigClass:
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_layers: int
    n_ctx: int
    eps: float
    d_vocab: int
    act_fn: str
    normalization_type: str
    architecture: str = None
    tokenizer_name: str = None
    n_key_value_heads: int = None
    is_low_precision: bool = False
    attn_types: list = None
    use_attn_scale: bool = None
    attn_scale: float = None
    use_local_attn: bool = None
    window_size: Optional[int] = None
    scale_attn_by_inverse_layer_idx: bool = None
    parallel_attn_mlp: bool = False
    pre_layernorm: bool = True
    post_layernorm: bool = False
    positional_embedding_type: str = "standard"
    rotary_dim: Optional[int] = None
    rotary_base: Optional[int] = None
    final_rms: bool = False
    attn_scores_soft_cap: int = None
    output_logits_soft_cap: int = None
    gated_mlp: bool = False
    model_type: str = "causal"
    model_modality: str = "language" # language, vision, (maybe "speech" one day?)
    label2id: Optional[Dict[str, int]] = None # for vision transformers
    id2label: Optional[Dict[int, str]] = None
    image_size: int = 224
    device: None | str | torch.device = None

@dataclass
class ModelMapData:
    config_map: Callable[[Any], MapConfigClass]
    model_map_dict: Dict[str, str]
    layer_map_factory: Callable
