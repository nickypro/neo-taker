## Transformer Activation taKER.

Minimal implementation of TransformerLens syntax with limited feature set, but allows using hooked HuggingFace models.

New model architectures must manually be mapped in ./component_maps/ but an LLM should be able to assist you if you don't already have access to it.

Additionally, some activations cannot be modified (eg: attention scores) because these do not have natural hook points in HuggingFace transformers. The current minimal implementation may have slightly unpredictable edits (eg: resid_mid only modifies inputs to mlp, the edits do not propagate.)

Example syntax:
```
from neo_taker import Model
m = Model(model_repo="nickypro/tinyllama-15m", model_device="cuda:0", dtype="bf16")
tokens = m.to_tokens("Hello world!")

print( m.list_activation_points() )

hook_fn_print_name = lambda act, hook: print(hook.name, act.shape)
with m.hooks(["blocks.0.hook_resid_post", hook_fn_print_name]):
    logits = m(tokens, return_type="logits")

```

See development repo here: [github.com/nickypro/neo-taker](https://github.com/nickypro/neo-taker)