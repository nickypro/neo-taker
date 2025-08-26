"""Streamlined Neo-Taker Model with TransformerLens compatibility."""

from typing import List, Optional, Union
import warnings
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig

from .data_classes import DtypeMap
from .hooks import HookedRootModule, HookPoint
from .component_maps import MapConfigClass, ModelMap, convert_hf_model_config

class Model(HookedRootModule):
    """Neo-Taker model with TransformerLens-style hooks."""
    
    def __init__(
        self,
        model_repo: str = "nickypro/tinyllama-15m",
        model_device: str = None,
        dtype: str = "bfp16",
        tokenizer_repo: str = None,
        eval_mode: bool = True,
        model_kwargs: dict = None,
    ):
        super().__init__()
        
        # Core configuration
        self.model_repo = model_repo
        self.tokenizer_repo = tokenizer_repo or model_repo
        self.eval_mode = eval_mode
        self.model_kwargs = model_kwargs or {}
        
        # Dtype handling
        self.dtype_map = DtypeMap(dtype)
        self.dtype = self.dtype_map._dtype
        self.dtype_args = self.dtype_map._dtype_args
        
        # Model components
        self.hf_config: AutoConfig = None
        self.cfg: MapConfigClass = None
        self.tokenizer: AutoTokenizer = None
        self.predictor: AutoModelForCausalLM = None
        self.model: PreTrainedModel = None
        self.map: ModelMap = None
        self.layers: list = None
        # Track which TL-style hook points are actually wired into the forward pass
        self._wired_hook_names: set[str] = set()

        # Initialize the model
        self._init_model()
        self.cfg.device = model_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._add_hook_points()
        self.setup()  # Setup TransformerLens hooks
        
        if self.eval_mode and self.predictor:
            self.predictor.eval()
    
    def _init_model(self):
        """Initialize the model components."""
        # Load configuration
        self.cfg, self.hf_config = convert_hf_model_config(self.model_repo)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_repo, 
            legacy=False, 
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_args = {**self.dtype_args, **self.model_kwargs}
        self.predictor = AutoModelForCausalLM.from_pretrained(
            self.model_repo,
            device_map="auto",
            **model_args,
        )
        
        # Create model mapping
        self.map = ModelMap(self.predictor, self.cfg)
        self.model = self.map["model"]
        self.layers = self.map.layers
        
        print(f"Loaded model '{self.model_repo}' with {self.dtype_map.str_dtype}")
    
    def _add_hook_points(self):
        """Add TransformerLens-style hook points and integrate them into model components."""
        # Embedding hook
        self.hook_embed = HookPoint()
        
        # Layer-specific hooks
        for i in range(self.cfg.n_layers):
            layer = self.layers[i]
            
            # Residual stream hooks
            setattr(self, f'blocks.{i}.hook_resid_pre', HookPoint())
            setattr(self, f'blocks.{i}.hook_resid_mid', HookPoint())
            setattr(self, f'blocks.{i}.hook_resid_post', HookPoint())
            
            # Layer norm hooks (monitoring scale and normalized output)
            setattr(self, f'blocks.{i}.ln1.hook_scale', HookPoint())
            setattr(self, f'blocks.{i}.ln1.hook_normalized', HookPoint())
            setattr(self, f'blocks.{i}.ln2.hook_scale', HookPoint())
            setattr(self, f'blocks.{i}.ln2.hook_normalized', HookPoint())
            
            # Attention hooks (monitoring intermediate values)
            setattr(self, f'blocks.{i}.attn.hook_k', HookPoint())       # After K projection
            setattr(self, f'blocks.{i}.attn.hook_q', HookPoint())       # After Q projection  
            setattr(self, f'blocks.{i}.attn.hook_v', HookPoint())       # After V projection
            setattr(self, f'blocks.{i}.attn.hook_z', HookPoint())       # After attention (pre-output proj)
            setattr(self, f'blocks.{i}.attn.hook_attn_scores', HookPoint())  # Raw attention scores
            setattr(self, f'blocks.{i}.attn.hook_pattern', HookPoint())      # Attention weights (post-softmax)
            setattr(self, f'blocks.{i}.attn.hook_result', HookPoint())       # Attention output per head
            setattr(self, f'blocks.{i}.attn.hook_rot_k', HookPoint())   # After rotary embedding on K
            setattr(self, f'blocks.{i}.attn.hook_rot_q', HookPoint())   # After rotary embedding on Q
            
            # MLP hooks (monitoring inputs/outputs)
            setattr(self, f'blocks.{i}.mlp.hook_pre', HookPoint())           # MLP input
            setattr(self, f'blocks.{i}.mlp.hook_pre_linear', HookPoint())    # After first linear layer
            setattr(self, f'blocks.{i}.mlp.hook_post', HookPoint())          # MLP output
            
            # Input/output hooks (monitoring what goes into/out of components)
            setattr(self, f'blocks.{i}.hook_attn_in', HookPoint())      # Input to attention
            setattr(self, f'blocks.{i}.hook_q_input', HookPoint())      # Input to Q projection
            setattr(self, f'blocks.{i}.hook_k_input', HookPoint())      # Input to K projection
            setattr(self, f'blocks.{i}.hook_v_input', HookPoint())      # Input to V projection
            setattr(self, f'blocks.{i}.hook_mlp_in', HookPoint())       # Input to MLP
            setattr(self, f'blocks.{i}.hook_attn_out', HookPoint())     # Output from attention
            setattr(self, f'blocks.{i}.hook_mlp_out', HookPoint())      # Output from MLP
        
        # Final layer norm hooks
        setattr(self, 'ln_final.hook_scale', HookPoint())
        setattr(self, 'ln_final.hook_normalized', HookPoint())
        
        # Now integrate hooks into the actual model forward pass
        self._integrate_hooks_into_model()
    
    def _integrate_hooks_into_model(self):
        """Integrate hook points using PyTorch's native hook system."""
        
        # Register PyTorch hooks that call our TransformerLens-style hooks
        
        # Hook the embedding layer
        embed_module = self.map["embed"]
        embed_module.register_forward_hook(self._create_pytorch_hook("hook_embed"))
        self._wired_hook_names.add("hook_embed")
        
        # Hook each transformer layer
        for i in range(self.cfg.n_layers):
            self._register_layer_hooks(i)
        
        # Hook final layer norm if it exists
        if "ln_final" in self.map.key_map:
            ln_final = self.map["ln_final"]
            # Hook input (scale) and output (normalized)
            ln_final.register_forward_pre_hook(self._create_pytorch_pre_hook(f"ln_final.hook_scale"))
            ln_final.register_forward_hook(self._create_pytorch_hook(f"ln_final.hook_normalized"))
            self._wired_hook_names.add("ln_final.hook_scale")
            self._wired_hook_names.add("ln_final.hook_normalized")
    
    def _create_pytorch_hook(self, hook_name):
        """Create a PyTorch forward hook that calls our TransformerLens hook."""
        def pytorch_hook(module, input, output):
            # Get our hook point
            hook_point = getattr(self, hook_name)
            # Call it with the output (TransformerLens style)
            return hook_point(output)
        return pytorch_hook
    
    def _create_pytorch_pre_hook(self, hook_name):
        """Create a PyTorch pre-forward hook that calls our TransformerLens hook."""
        def pytorch_pre_hook(module, input):
            # Get our hook point
            hook_point = getattr(self, hook_name)
            # Call it with the input (TransformerLens style)
            if isinstance(input, tuple) and len(input) > 0:
                modified_input = hook_point(input[0])
                if len(input) > 1:
                    return (modified_input,) + input[1:]
                else:
                    return (modified_input,)
            elif isinstance(input, Tensor):
                return hook_point(input)
            else:
                # If input is not a tensor or tuple, just return it unchanged
                return input
        return pytorch_pre_hook
    
    def _register_layer_hooks(self, layer_idx):
        """Register PyTorch hooks for a specific transformer layer."""
        layer = self.layers[layer_idx]
        
        # Get the actual HuggingFace transformer layer
        hf_layer = layer.layer
        
        # Hook residual stream at layer level
        hf_layer.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.hook_resid_pre"))
        def resid_post_hook(module, inp, out):
            hp = getattr(self, f"blocks.{layer_idx}.hook_resid_post")
            if isinstance(out, tuple) and len(out) > 0:
                t0 = hp(out[0])
                return (t0,) + out[1:]
            return hp(out)
        hf_layer.register_forward_hook(resid_post_hook)
        self._wired_hook_names.add(f"blocks.{layer_idx}.hook_resid_pre")
        self._wired_hook_names.add(f"blocks.{layer_idx}.hook_resid_post")
        
        # Try to hook attention and MLP components if they exist
        try:
            # Hook attention components (support both LLaMA `self_attn` and GPT-2 `attn`)
            if hasattr(hf_layer, 'self_attn') or hasattr(hf_layer, 'attn'):
                attn = hf_layer.self_attn if hasattr(hf_layer, 'self_attn') else hf_layer.attn

                # Hook attention input
                attn.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.hook_attn_in"))

                # For attn_out, the module often returns a tuple; pass only tensor[0] to TL hook
                def attn_out_hook(module, inp, out):
                    hp = getattr(self, f"blocks.{layer_idx}.hook_attn_out")
                    if isinstance(out, tuple) and len(out) > 0:
                        t0 = hp(out[0])
                        return (t0,) + out[1:]
                    return hp(out)
                attn.register_forward_hook(attn_out_hook)
                self._wired_hook_names.add(f"blocks.{layer_idx}.hook_attn_in")
                self._wired_hook_names.add(f"blocks.{layer_idx}.hook_attn_out")

                # Helper to compute head counts
                def _num_heads(which: str) -> int:
                    n_qo_heads = self.cfg.n_heads
                    n_kv_heads = getattr(self.cfg, 'n_key_value_heads', None) or n_qo_heads
                    return n_kv_heads if which in ('k', 'v') else n_qo_heads

                d_head = self.cfg.d_head

                # Custom forward hook for Q/K/V to expose [batch, seq, n_heads, d_head] to hooks
                def _qkv_hook(which: str):
                    def _hook(module, inp, out):
                        hp = getattr(self, f"blocks.{layer_idx}.attn.hook_{which}")
                        if not isinstance(out, torch.Tensor) or out.dim() != 3:
                            return hp(out)
                        B, S, D = out.shape
                        # Heads per stream
                        nH_qo = _num_heads('q')
                        nH_kv = _num_heads('k')
                        d_model_qo = nH_qo * d_head
                        d_model_kv = nH_kv * d_head

                        # Case 1: separate projection with expected dimension
                        nH = _num_heads(which)
                        expected = nH * d_head
                        if D == expected:
                            t = out.view(B, S, nH, d_head)
                            t2 = hp(t)
                            if t2 is None:
                                t2 = t
                            return t2.reshape(B, S, expected)

                        # Case 2: combined QKV (e.g., GPT-2 c_attn): split, apply, and re-concat
                        if D == (d_model_qo + d_model_kv + d_model_kv):
                            q_end = d_model_qo
                            k_end = q_end + d_model_kv
                            q_seg = out[..., :q_end]
                            k_seg = out[..., q_end:k_end]
                            v_seg = out[..., k_end:]

                            if which == 'q':
                                t = q_seg.view(B, S, nH_qo, d_head)
                                t2 = hp(t)
                                if t2 is None:
                                    t2 = t
                                q_seg = t2.reshape(B, S, d_model_qo)
                            elif which == 'k':
                                t = k_seg.view(B, S, nH_kv, d_head)
                                t2 = hp(t)
                                if t2 is None:
                                    t2 = t
                                k_seg = t2.reshape(B, S, d_model_kv)
                            elif which == 'v':
                                t = v_seg.view(B, S, nH_kv, d_head)
                                t2 = hp(t)
                                if t2 is None:
                                    t2 = t
                                v_seg = t2.reshape(B, S, d_model_kv)
                            return torch.cat([q_seg, k_seg, v_seg], dim=-1)

                        # Fallback: warn and pass through hook without reshaping
                        warnings.warn(
                            f"Unexpected proj dim at layer {layer_idx} for {which}: {D}; cannot infer head split"
                        )
                        return hp(out)
                    return _hook

                # Hook Q, K, V projections using model mapping
                mod_to_whiches = {}
                proj_modules = {}
                if "q_proj" in layer.key_map:
                    proj_modules['q'] = layer["q_proj"]
                    self._wired_hook_names.add(f"blocks.{layer_idx}.hook_q_input")
                    self._wired_hook_names.add(f"blocks.{layer_idx}.attn.hook_q")
                if "k_proj" in layer.key_map:
                    proj_modules['k'] = layer["k_proj"]
                    self._wired_hook_names.add(f"blocks.{layer_idx}.hook_k_input")
                    self._wired_hook_names.add(f"blocks.{layer_idx}.attn.hook_k")
                if "v_proj" in layer.key_map:
                    proj_modules['v'] = layer["v_proj"]
                    self._wired_hook_names.add(f"blocks.{layer_idx}.hook_v_input")
                    self._wired_hook_names.add(f"blocks.{layer_idx}.attn.hook_v")

                # Group whiches by underlying module (to deduplicate pre-hooks for combined QKV)
                for which, mod in proj_modules.items():
                    mod_to_whiches.setdefault(id(mod), {"module": mod, "whiches": []})["whiches"].append(which)

                # Register a single pre-hook per unique module that chains q/k/v input hooks
                for entry in mod_to_whiches.values():
                    mod = entry["module"]
                    whiches = entry["whiches"]
                    def _chained_prehook(module, input, whiches=whiches):
                        hp_inputs = {
                            'q': getattr(self, f"blocks.{layer_idx}.hook_q_input"),
                            'k': getattr(self, f"blocks.{layer_idx}.hook_k_input"),
                            'v': getattr(self, f"blocks.{layer_idx}.hook_v_input"),
                        }
                        if isinstance(input, tuple) and len(input) > 0:
                            x = input[0]
                            for w in whiches:
                                x2 = hp_inputs[w](x)
                                if x2 is not None:
                                    x = x2
                            return (x,) + input[1:]
                        elif isinstance(input, Tensor):
                            x = input
                            for w in whiches:
                                x2 = hp_inputs[w](x)
                                if x2 is not None:
                                    x = x2
                            return x
                        return input
                    mod.register_forward_pre_hook(_chained_prehook)

                # Register forward hooks (one per which) to expose per-head Q/K/V
                for which, mod in proj_modules.items():
                    mod.register_forward_hook(_qkv_hook(which))

                # Custom pre-hook for Z to expose [batch, seq, n_heads, d_head] to hooks
                def _z_prehook(module, input):
                    hp = getattr(self, f"blocks.{layer_idx}.attn.hook_z")
                    if not (isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor)):
                        return input
                    x = input[0]
                    if x.dim() != 3:
                        return input
                    B, S, D = x.shape
                    nH = _num_heads('q')  # z uses qo heads
                    expected = nH * d_head
                    if D != expected:
                        warnings.warn(f"Unexpected z dim: {D} != n_heads({nH})*d_head({d_head}) at layer {layer_idx}")
                        t2 = hp(x)
                        return (t2,) + input[1:] if len(input) > 1 else (t2,)
                    t = x.view(B, S, nH, d_head)
                    t2 = hp(t)
                    if t2 is None:
                        t2 = t
                    flat = t2.reshape(B, S, expected)
                    if len(input) > 1:
                        return (flat,) + input[1:]
                    else:
                        return (flat,)

                if "o_proj" in layer.key_map:
                    out_proj = layer["o_proj"]
                    out_proj.register_forward_pre_hook(_z_prehook)
                    out_proj.register_forward_hook(self._create_pytorch_hook(f"blocks.{layer_idx}.hook_attn_out"))
                    self._wired_hook_names.add(f"blocks.{layer_idx}.attn.hook_z")
                    self._wired_hook_names.add(f"blocks.{layer_idx}.hook_attn_out")
            
            # Hook MLP components
            if hasattr(hf_layer, 'mlp'):
                mlp = hf_layer.mlp
                
                # Hook MLP input/output
                mlp.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.hook_mlp_in"))
                # Wire resid_mid at MLP input (residual after attention)
                mlp.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.hook_resid_mid"))
                mlp.register_forward_hook(self._create_pytorch_hook(f"blocks.{layer_idx}.hook_mlp_out"))
                self._wired_hook_names.add(f"blocks.{layer_idx}.hook_mlp_in")
                self._wired_hook_names.add(f"blocks.{layer_idx}.hook_resid_mid")
                self._wired_hook_names.add(f"blocks.{layer_idx}.hook_mlp_out")
                
                # Hook MLP components
                mlp.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.mlp.hook_pre"))
                mlp.register_forward_hook(self._create_pytorch_hook(f"blocks.{layer_idx}.mlp.hook_post"))
                self._wired_hook_names.add(f"blocks.{layer_idx}.mlp.hook_pre")
                self._wired_hook_names.add(f"blocks.{layer_idx}.mlp.hook_post")
            
            # Hook layer norms using model mapping
            # Use TransformerLens naming: ln1 and ln2 via map only
            if "ln1" in layer.key_map:
                ln1 = layer["ln1"]
                ln1.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.ln1.hook_scale"))
                ln1.register_forward_hook(self._create_pytorch_hook(f"blocks.{layer_idx}.ln1.hook_normalized"))
                self._wired_hook_names.add(f"blocks.{layer_idx}.ln1.hook_scale")
                self._wired_hook_names.add(f"blocks.{layer_idx}.ln1.hook_normalized")
            if "ln2" in layer.key_map:
                ln2 = layer["ln2"]
                ln2.register_forward_pre_hook(self._create_pytorch_pre_hook(f"blocks.{layer_idx}.ln2.hook_scale"))
                ln2.register_forward_hook(self._create_pytorch_hook(f"blocks.{layer_idx}.ln2.hook_normalized"))
                self._wired_hook_names.add(f"blocks.{layer_idx}.ln2.hook_scale")
                self._wired_hook_names.add(f"blocks.{layer_idx}.ln2.hook_normalized")

        except Exception as e:
            print(f"Warning: Could not register all hooks for layer {layer_idx}: {e}")

    def get_activation_map(self, wired_only: bool = True) -> dict[str, HookPoint]:
        """Return a map of TL-style activation hook names to HookPoint modules.

        If wired_only is True, only returns hook points that are actually integrated into the
        forward pass (excludes non-existent points like attn_scores/pattern where unavailable).
        """
        all_points = self.hook_points()
        if wired_only:
            return {name: hp for name, hp in all_points.items() if name in self._wired_hook_names}
        return all_points

    def list_activation_points(self, wired_only: bool = True) -> List[str]:
        """List available TL-style activation hook names.

        If wired_only is True, only lists those actually connected to real activations.
        """
        return sorted(self.get_activation_map(wired_only=wired_only).keys())
    
    def forward(
        self, 
        tokens=None, 
        input_ids=None,
        return_type="logits",
        **kwargs
    ):
        """Forward pass through the model with TransformerLens-style interface."""
        # Handle input
        if tokens is not None:
            input_ids = tokens
        elif input_ids is None:
            raise ValueError("Must provide either 'tokens' or 'input_ids'")
            
        # Ensure tokens are on the right device
        if hasattr(input_ids, 'to'):
            input_ids = input_ids.to(next(self.predictor.parameters()).device)
        
        # Generate output based on return_type
        if return_type == "logits":
            return self.get_logits(input_ids=input_ids, **kwargs)
        elif return_type == "loss":
            logits = self.get_logits(input_ids=input_ids, **kwargs)
            targets = input_ids[:, 1:]
            logits = logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1),
                ignore_index=-100
            )
            return loss
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
    
    def __call__(self, *args, **kwargs):
        """Make the model callable."""
        return self.forward(*args, **kwargs)
    
    # Core model functionality
    def get_logits(self, input_ids=None, text=None, **kwargs):
        """Get logits from input."""
        if text is not None:
            input_ids = self.to_tokens(text)
        
        # Ensure tokens are on the right device
        if hasattr(input_ids, 'to'):
            input_ids = input_ids.to(next(self.predictor.parameters()).device)
        
        outputs = self.predictor(input_ids, **kwargs)
        return outputs.logits
    
    def generate(self, text=None, input_ids=None, tokens=None, max_new_tokens=10, **kwargs):
        """Generate token ids (TransformerLens-compatible).

        Accepts `text` (str or list[str]) or pre-tokenized `tokens` / `input_ids` (Tensor).
        Returns the full generated token ids, including the prompt, on the model device.
        """
        # Normalize inputs
        if tokens is not None:
            input_ids = tokens
        elif input_ids is None and text is not None:
            input_ids = self.to_tokens(text)
        elif input_ids is None and isinstance(text, Tensor):
            input_ids = text

        if not isinstance(input_ids, Tensor):
            raise ValueError("generate expects `text` (str/list[str]) or `tokens`/`input_ids` (Tensor)")

        # Ensure tokens are on model device
        device = next(self.predictor.parameters()).device
        input_ids = input_ids.to(device)

        # Ensure a pad token id for HF generate
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            generated = self.predictor.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                **kwargs,
            )

        return generated

    def generate_text(self, *args, **kwargs) -> list[str]:
        """Convenience: decode generated tokens to strings."""
        generated = self.generate(*args, **kwargs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    # TransformerLens-style tokenizer utilities
    def to_tokens(self, text, prepend_bos=True):
        """Convert text (str or list[str]) to token ids Tensor [batch, seq]."""
        # If already a tensor, assume it's tokens and return as batch
        if isinstance(text, Tensor):
            return text if text.dim() == 2 else text.unsqueeze(0)

        if isinstance(text, str):
            text = [text]
        
        encoded_list = []
        for t in text:
            if prepend_bos and getattr(self.tokenizer, 'bos_token', None):
                t = (self.tokenizer.bos_token or "") + t
            encoded = self.tokenizer.encode(t, return_tensors="pt", add_special_tokens=False)
            encoded_list.append(encoded.squeeze(0))
        
        # Pad to same length
        if len(encoded_list) > 1:
            max_len = max(t.size(0) for t in encoded_list)
            padded = []
            for t in encoded_list:
                if t.size(0) < max_len:
                    pad_length = max_len - t.size(0)
                    pad_val = self.tokenizer.pad_token_id
                    if pad_val is None:
                        pad_val = self.tokenizer.eos_token_id
                    t = torch.cat([t, torch.full((pad_length,), pad_val, dtype=t.dtype)])
                padded.append(t)
            tokens = torch.stack(padded)
        else:
            tokens = encoded_list[0].unsqueeze(0)
        return tokens.to(self.cfg.device)
    
    def to_string(self, tokens):
        """Convert tokens to string(s).

        - If `tokens` is shape [seq], returns a single string.
        - If `tokens` is shape [batch, seq], returns a list[str] (one per batch element).
        - Also accepts list[int] or list[list[int]].
        """
        # Normalize to CPU and python lists
        if isinstance(tokens, Tensor):
            if tokens.dim() == 1:
                ids = tokens.detach().cpu().tolist()
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            elif tokens.dim() == 2:
                batch_ids = [row.detach().cpu().tolist() for row in tokens]
                return self.tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
            else:
                raise ValueError("tokens must be a 1D or 2D Tensor")

        # Handle python lists
        if isinstance(tokens, list):
            if len(tokens) == 0:
                return ""
            # list of list[int]
            if isinstance(tokens[0], list):
                return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            # list[int]
            else:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)

        raise ValueError("Unsupported tokens type for to_string")
    
    def to_str_tokens(self, tokens, pretty: bool = True, remove_special: bool = False):
        """Convert token ids to a list of string tokens (optionally pretty-print).

        pretty=True will map common BPE/SentencePiece markers (eg "Ġ" -> space, "Ċ" -> \n, "▁" -> space).
        remove_special=True will drop tokens that tokenizer flags as special tokens.
        """
        # Normalize to batch list of lists
        if isinstance(tokens, Tensor):
            if tokens.dim() == 1:
                token_id_seqs = [tokens.tolist()]
            elif tokens.dim() == 2:
                token_id_seqs = [row.tolist() for row in tokens]
            else:
                raise ValueError("tokens must be 1D or 2D Tensor")
        else:
            # Assume list of ids or list of list of ids
            if len(tokens) > 0 and isinstance(tokens[0], list):
                token_id_seqs = tokens
            else:
                token_id_seqs = [tokens]

        out: list[list[str]] = []
        for ids in token_id_seqs:
            raw_tokens = self.tokenizer.convert_ids_to_tokens(ids)

            if remove_special and hasattr(self.tokenizer, "all_special_tokens"):
                specials = set(self.tokenizer.all_special_tokens)
            else:
                specials = set()

            if not pretty:
                out.append([tok for tok in raw_tokens if tok not in specials])
                continue

            pretty_tokens: list[str] = []
            for i, tok in enumerate(raw_tokens):
                if tok in specials:
                    if not remove_special:
                        pretty_tokens.append(tok)
                    continue

                # GPT-2 style byte-level BPE markers
                if tok.startswith("Ġ"):
                    tok = " " + tok[1:]
                tok = tok.replace("Ċ", "\n")

                # SentencePiece marker for space
                if tok.startswith("▁"):
                    # add a space if not the first token
                    tok = (" " if len(pretty_tokens) > 0 else "") + tok.replace("▁", "", 1)

                pretty_tokens.append(tok)

            out.append(pretty_tokens)

        return out if len(out) > 1 else out[0]

    def to_single_token(self, string: str) -> int:
        """Map a string that is exactly one token to its token id.

        Raises an error if the input string does not correspond to a single token.
        """
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        try: 
            assert not token.shape, f"Input string: {string} is not a single token!"
            return token.item()
        except AssertionError:
            print(f"WARNING: Input string: {string} is not a single token! Got {self.to_str_tokens(token)}")
            token = token[-1]
            return token.item()

    # Utility methods
    def to(self, device):
        """Move model to device."""
        self.cfg.device = device
        if hasattr(self, 'predictor') and self.predictor:
            self.predictor.to(device)
        return self

    def show_details(self, verbose=True):
        """Show model details."""
        if verbose:
            print(f" - n_layers : {self.cfg.n_layers}")
            print(f" - d_model  : {self.cfg.d_model}")
            print(f" - n_heads  : {self.cfg.n_heads}")
            print(f" - d_head   : {self.cfg.d_head}")
            print(f" - d_mlp    : {self.cfg.d_mlp}")
        else:
            print(f" - n_layers, d_model = {self.cfg.n_layers}, {self.cfg.d_model}")
