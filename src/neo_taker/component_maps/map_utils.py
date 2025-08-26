
# General Helper Functions
#####################################################################################

def get_attrs(obj, attr_string):
    nested_attributes = attr_string.split('.')
    current_attr = obj
    for attr_name in nested_attributes:
        current_attr = getattr(current_attr, attr_name)
    return current_attr

def set_attrs(obj, attr_string, value, override=True):
    nested_attrs = attr_string.split('.')
    nested_attrs, final_attr = nested_attrs[:-1], nested_attrs[-1]
    current_attr = get_attrs(obj, ".".join(nested_attrs)) if len(nested_attrs) > 0 else obj
    if not override and hasattr(current_attr, final_attr):
        return
    setattr(current_attr, final_attr, value)

# Architecture Map Helper Functions
#####################################################################################

def generate_attn_qkv_functions(weight_fn, bias_fn):
    return {
        "attn.W_Q"  : lambda layer, inpt=None: weight_fn(layer, "q", inpt),
        "attn.W_K"  : lambda layer, inpt=None: weight_fn(layer, "k", inpt),
        "attn.W_V"  : lambda layer, inpt=None: weight_fn(layer, "v", inpt),
        "attn.b_Q"  : lambda layer, inpt=None: bias_fn(layer, "q", inpt),
        "attn.b_K"  : lambda layer, inpt=None: bias_fn(layer, "k", inpt),
        "attn.b_V"  : lambda layer, inpt=None: bias_fn(layer, "v", inpt),
    }

def update_param(module, param_key, new_param):
    params = module.state_dict()
    assert param_key in params
    params[param_key] = new_param
    module.load_state_dict(params)

def generate_sizes_dict(einops_str, cfg):
    sizes_dict = {}
    if "qkv" in einops_str:
        sizes_dict["qkv"] = 3
    if "d_head" in einops_str:
        sizes_dict["d_head"] = cfg.d_head
    if "n_heads" in einops_str:
        sizes_dict["n_heads"] = cfg.n_heads
    if "d_model" in einops_str:
        sizes_dict["d_model"] = cfg.d_model
    if "d_mlp" in einops_str:
        sizes_dict["d_mlp"] = cfg.d_mlp
    if "n_layers" in einops_str:
        sizes_dict["n_layers"] = cfg.n_layers
    return sizes_dict
