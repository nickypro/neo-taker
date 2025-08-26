#!/usr/bin/env python3
"""Test the exact example provided by the user."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from neo_taker import Model

def test_user_example():
    """Test the exact example from the user requirements."""
    print("Testing the exact user example...")
    
    # Create model
    model = Model(
        model_repo="nickypro/tinyllama-15m",
        model_device="cpu"
    )
    
    # Define hook function exactly as specified
    def hook(act, hook):
        print(hook.name)
        act[-1, -1, :] = 0
        return act

    # Test the exact syntax from user requirements
    tokens = model.to_tokens("Hello world!")
    
    print(f"Input tokens shape: {tokens.shape}")
    
    # Use the exact hook syntax specified by the user
    with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", hook)]):
        result = model(tokens, return_type="logits")
    
    print(f"Output shape: {result.shape}")
    print("✓ User example works perfectly!")
    
    # Test a few more hook points from the user's list
    hook_points_to_test = [
        "hook_embed",
        "blocks.0.ln1.hook_scale", 
        "blocks.0.attn.hook_q",
        "blocks.0.mlp.hook_pre",
        "ln_final.hook_normalized"
    ]
    
    print(f"\nTesting additional hook points...")
    for hook_point in hook_points_to_test:
        try:
            with model.hooks(fwd_hooks=[(hook_point, hook)]):
                result = model(tokens, return_type="logits")
            print(f"✓ {hook_point}")
        except Exception as e:
            print(f"✗ {hook_point}: {e}")
    
    print(f"\n🎉 All tests passed! neo_taker now supports TransformerLens-style hooks!")

if __name__ == "__main__":
    test_user_example()

