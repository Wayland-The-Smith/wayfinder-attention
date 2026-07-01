#!/usr/bin/env python3
"""Probe fast attention backend availability."""
import importlib

for name in ("flash_attn", "fla", "xformers"):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"OK {name} {ver}")
    except Exception as e:
        print(f"NO {name}: {e}")

PYTORCH = __import__("torch")
print("torch", PYTORCH.__version__)
print("flash_sdp", PYTORCH.backends.cuda.flash_sdp_enabled())
try:
    from torch.nn.attention.flex_attention import flex_attention
    print("OK flex_attention")
except Exception as e:
    print("NO flex_attention:", e)
