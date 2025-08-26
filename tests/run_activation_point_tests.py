#!/usr/bin/env python3
"""
Validate all wired TransformerLens-style activation points:
- Capture activations and verify shapes
- Zero activations and measure loss delta
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from neo_taker import Model


def main(model_repo):
    torch.set_grad_enabled(False)

    # Build model on CPU for speed/stability
    model = Model(model_repo=model_repo, model_device="cpu", eval_mode=True)

    # Prepare a short prompt
    text = "Hello world! This is a tiny test."
    tokens = model.to_tokens(text)

    # Get baseline loss
    baseline_loss = model(tokens, return_type="loss").item()

    # Collect wired activation points
    hook_names = model.list_activation_points(wired_only=True)

    results = []

    # 1) Capture activations and verify shapes
    for name in hook_names:
        captured = {}

        def save_hook(act, hook):
            captured["act"] = act.detach().clone()
            return act

        try:
            with model.hooks(fwd_hooks=[(name, save_hook)]):
                _ = model(tokens, return_type="logits")
        except Exception as e:
            results.append({
                "name": name,
                "ok": False,
                "error": f"capture_error: {e}",
            })
            continue

        ok = True
        err = None
        shape = None
        if "act" not in captured:
            ok = False
            err = "no_activation_captured"
        else:
            act = captured["act"]
            shape = tuple(act.shape)
            # For Q/K/V/Z, we now expose per-head views: [batch, seq, n_heads, d_head]
            if any(name.endswith(suf) for suf in [".attn.hook_q", ".attn.hook_k", ".attn.hook_v", ".attn.hook_z"]):
                if act.dim() != 4:
                    ok = False
                    err = f"bad_rank:{act.dim()} (expected 4 for per-head)"
                else:
                    b, s, nH, dH = act.shape
                    if b != tokens.shape[0]:
                        ok = False
                        err = f"bad_batch:{b}"
                    if s != tokens.shape[1]:
                        ok = False
                        err = f"bad_seq:{s}"
                    # Optional: sanity check head dims multiply to d_model
                    # Can't know flatten dim here, but can compare to cfg
                    from neo_taker import Model as _M
                    # Using the outer scope model
                    n_qo = model.cfg.n_heads
                    n_kv = getattr(model.cfg, 'n_key_value_heads', None) or n_qo
                    exp_heads = n_kv if name.endswith(('.attn.hook_k', '.attn.hook_v')) else n_qo
                    if nH != exp_heads or dH != model.cfg.d_head:
                        ok = False
                        err = f"bad_head_shape:{(nH,dH)} exp:{(exp_heads, model.cfg.d_head)}"
            else:
                # Expect 3D: [batch, seq, hidden]
                if act.dim() != 3:
                    ok = False
                    err = f"bad_rank:{act.dim()}"
                else:
                    if act.shape[0] != tokens.shape[0]:
                        ok = False
                        err = f"bad_batch:{act.shape[0]}"
                    if act.shape[1] != tokens.shape[1]:
                        ok = False
                        err = f"bad_seq:{act.shape[1]}"

        results.append({
            "name": name,
            "ok": ok,
            "shape": shape,
            "error": err,
        })

    # 2) Zero activations and measure loss delta
    loss_deltas = []
    for name in hook_names:
        def zero_hook(act, hook):
            return torch.zeros_like(act)

        try:
            with model.hooks(fwd_hooks=[(name, zero_hook)]):
                zero_loss = model(tokens, return_type="loss").item()
        except Exception as e:
            loss_deltas.append({
                "name": name,
                "ok": False,
                "error": f"zero_error: {e}",
            })
            continue

        loss_deltas.append({
            "name": name,
            "ok": True,
            "baseline": baseline_loss,
            "zero_loss": zero_loss,
            "delta": zero_loss - baseline_loss,
            "improved": zero_loss < baseline_loss,
        })

    # Print summary
    print("\n=== Activation Capture Results ===")
    for r in results:
        if r["ok"]:
            print(f"[OK] {r['name']}: shape={r['shape']}")
        else:
            print(f"[FAIL] {r['name']}: {r['error']}")

    print("\nBaseline loss:", baseline_loss)
    print("\n=== Zeroing Loss Deltas (delta = zero_loss - baseline) ===")
    for r in loss_deltas:
        if r["ok"]:
            tag = "↓" if r["improved"] else "↑"
            print(f"{tag} {r['name']}: delta={r['delta']:.4f} (zero={r['zero_loss']:.4f})")
        else:
            print(f"[FAIL] {r['name']}: {r['error']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_activation_point_tests.py [model_repo]")
        model_repo = "nickypro/tinyllama-15m"
    else:
        model_repo = sys.argv[1]

    main(model_repo)

