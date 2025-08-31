# Code mostly from TransformerLens
# https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
import types
import copy
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import einops
from transformers import AutoConfig
import torch

from .map_utils import get_attrs, set_attrs
from .map_data_classes import MapConfigClass

from .modelling_llama import LlamaModelMapData
from .modelling_gpt2 import GPT2ModelMapData
from .modelling_gemma3 import Gemma3ModelMapData

def convert_hf_model_config(official_model_name: str):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the fig format.

    Takes the official_model_name as an input.
    """
    # Load HuggingFace model config
    #if 'llama' in official_model_name and 'open_llama' not in official_model_name:
    #    architecture = "LLaMAForCausalLM"
    #else:
    hf_config = AutoConfig.from_pretrained(official_model_name)
    architecture = hf_config.architectures[0]

    if architecture == "LlamaForCausalLM":
        cfg = LlamaModelMapData.config_map(hf_config)
    elif architecture == "GPT2LMHeadModel":
        cfg = GPT2ModelMapData.config_map(hf_config)
    elif architecture == "Gemma3ForCausalLM":
        cfg = Gemma3ModelMapData.config_map(hf_config)
    # elif architecture == "MistralForCausalLM":
    #     cfg_dict = MistralModelMapData.config_map(hf_config)
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    
    # All of these models use LayerNorm
    cfg.architecture = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg.tokenizer_name = official_model_name

    return cfg, hf_config


#####################################################################################
# Build Model Layer Map interfaces
#####################################################################################


def get_model_key_map(config: MapConfigClass):
    architecture = config.architecture
    if architecture in ["LLaMAForCausalLM", "LlamaForCausalLM"]:
        return LlamaModelMapData.model_map_dict
    elif architecture == "GPT2LMHeadModel":
        return GPT2ModelMapData.model_map_dict
    elif architecture == "Gemma3ForCausalLM":
        return Gemma3ModelMapData.model_map_dict
    # if architecture == "MistralForCausalLM":
    #     return mistral_model_map

    raise NotImplementedError(f"Architecture {architecture} not implemented")

def get_layer_key_map(config: MapConfigClass):
    architecture = config.architecture

    if architecture in ["LLaMAForCausalLM", "LlamaForCausalLM"]:
        return LlamaModelMapData.layer_map_factory(config)
    elif architecture == "GPT2LMHeadModel":
        return GPT2ModelMapData.layer_map_factory(config)
    elif architecture == "Gemma3ForCausalLM":
        return Gemma3ModelMapData.layer_map_factory(config)
    # if architecture == "MistralForCausalLM":
    #     return build_mistral_layer_map(config)

    raise NotImplementedError(f"Architecture {architecture} not implemented")


# Define Real Model Map and Layer Maps
######################################

class ModelMap:
    def __init__(self, model, cfg):
        self.cfg         = cfg
        self.model       = model
        self.key_map     = get_model_key_map(cfg)

        # Handle layers
        self.orig_layers = self["layers"]
        self.layers = [
            ModelLayerMap(self.cfg, layer) for layer in self.orig_layers
        ]

    def __getitem__(self, __name: str):
        key = self.key_map[__name]
        return get_attrs(self.model, key)

    def __setitem__(self, key, inpt):
        keys = key.split('.')
        attr = keys[-1]
        module = get_attrs(self.model, ".".join(keys[:-1]))
        params = module.state_dict()
        params[attr] = inpt
        module.load_state_dict(params)

class ModelLayerMap:
    def __init__(self, cfg, layer):
        self.cfg   = cfg
        self.layer = layer
        self.key_map = get_layer_key_map(cfg)

    @property
    def names(self):
        return list(self.key_map.keys())

    def __contains__(self, __name):
        return (__name in self.key_map)

    def __getitem__(self, __name):
        key = self.key_map[__name]

        if isinstance(key, str):
            if key == "layer":
                return self.layer
            if "inv_out_proj" in key:
                return NotImplementedError()
            return get_attrs(self.layer, key)

        if isinstance(key, Callable):
            return key(self.layer)

    def __setitem__(self, __name: str, __value: Any) -> None:
        key = self.key_map[__name]
        if isinstance(key, Callable):
            return key(self.layer, __value)

        if key is None:
            return None

        if not isinstance(key, str):
            raise ValueError("Invalid key, must be string or callable")

        # Get the module and attribute name
        keys = key.split('.')
        module = get_attrs(self.layer, ".".join(keys[:-1]))
        attr   = keys[-1]

        if attr == "inv_out_proj":
            setattr(module, attr, __value)
            return

        # If setting an attribute of a module (eg: weights or biases), update
        params = module.state_dict()
        params[attr] = __value
        module.load_state_dict(params)
        return

    def __str__(self):
        out_str  = "Wrapper for Transformer Layer\n"
        out_str += self.key_map.keys().__str__()
        out_str += "\nOriginal Layer Structure:\n"
        out_str += self.layer.__str__()
        return out_str

    def __getattr__(self, __name):
        key = self.key_map[__name]

        # Find all names that start with __name followed by a dot
        prefix = __name + "."
        remaining_names = sorted([n.split(prefix, 1)[1] for n in self.names if n.startswith(prefix)])

        if isinstance(key, str):
            mod = get_attrs(self.layer, key)
            for n in remaining_names:
                set_attrs(mod, n, self[f"{__name}.{n}"], override=False)
            return mod

        if isinstance(key, Callable):
            return key(self.layer)