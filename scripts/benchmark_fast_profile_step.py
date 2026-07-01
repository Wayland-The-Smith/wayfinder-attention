#!/usr/bin/env python3
"""Quick step-time estimate for fast suite profile."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, init_experiment_runtime, load_experiment_config
from experiments.experiment_7 import EXP7_ATTENTION_VARIANTS, _effective_train_batch_size
from experiments.experiment_4 import _apply_variant_config
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile
from routing_attention.utils.config import load_config, merge_configs

VARIANTS = [
    "dense_flash",
    "linear",
    "routing_asymmetric",
    "local_window64",
]


def bench_variant(cfg: dict, variant: str, T: int, device: torch.device) -> float:
    var_config = _apply_variant_config(cfg.copy(), variant)
    var_config["model"]["max_seq_len"] = T
    attn = var_config.get("model", {}).get("attention_type") or EXP7_ATTENTION_VARIANTS.get(variant, "routing")
    bs = _effective_train_batch_size(int(cfg["data"]["batch_size"]), T, attn)

    if attn == "routing":
        from experiments.common import load_router_from_reuse
        router, _ = load_router_from_reuse(var_config, device)
        model = build_transformer(var_config, attention_type="routing", router=router).to(device)
    else:
        model = build_transformer(var_config, attention_type=attn).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(1, 256, (bs, T), device=device)
    y = x.clone()
    y[:, : T // 2] = -100

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x)["logits"]
            loss = F.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)),
                y[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x)["logits"]
            loss = F.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)),
                y[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del model
    torch.cuda.empty_cache()
    return 1000.0 * sum(times) / len(times)


def _profile_config(profile: str) -> dict:
    raw = load_config(ROOT / "configs" / "experiment_7.yaml")
    patched = apply_suite_profile(raw, profile)
    ovr = {
        "model": patched.get("model", {}),
        "transformer": patched.get("transformer", {}),
        "data": patched.get("data", {}),
    }
    return merge_configs(load_experiment_config(7), ovr)


def main():
    for profile in ("fast", "full"):
        cfg = _profile_config(profile)
        device = init_experiment_runtime(cfg)
        n_layers = cfg["model"]["n_layers"]
        steps = cfg["transformer"]["max_steps"]
        print(f"\n=== profile={profile} n_layers={n_layers} max_steps={steps} ===")
        total_min = 0.0
        for T in (8192, 4096):
            for var in VARIANTS:
                ms = bench_variant(cfg, var, T, device)
                run_min = ms * steps / 1000 / 60
                total_min += run_min
                print(f"  T={T:5d} {var:22s} bs-cap  {ms:6.0f} ms/step  ~{run_min:5.1f} min/run")
        print(f"  (partial est. 4 variants × 2 lengths) subtotal ~{total_min:.0f} min")


if __name__ == "__main__":
    main()
