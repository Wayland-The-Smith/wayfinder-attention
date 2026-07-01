# Wayfinder Attention

*Attention as learned search: separate address-based retrieval from QKV aggregation.*

Official code release for [**Wayfinder Attention: Attention as Learned Search**](wayfinder_attention_paper_v0_4.tex) — a proof-cell study of learned-address sparse attention that distills dense teacher neighborhoods into a top-$K$ token index, then runs ordinary softmax over retrieved QKV.

**Paper (Zenodo):** [10.5281/zenodo.21087516](https://doi.org/10.5281/zenodo.21087516)

## Overview

Dense causal attention is brute-force search plus readout: each query scores the full prefix, then aggregates values through softmax. **Wayfinder Attention** asks whether those neighborhoods can be recovered by a **learned asymmetric address index** that retrieves a fixed top-$K$ set, delegating softmax to standard QKV — and whether **dedicated addresses** matter when the sparse forward budget is held fixed.

**Training recipe (3 phases):**

1. **Phase A** — dense FlashAttention teacher on needle-in-a-haystack (NIAH)
2. **Phase B** — multi-positive InfoNCE on teacher top-32 neighborhoods (retrieval *order*, not the full attention matrix)
3. **Phase C** — sparse top-$K{=}128$ fine-tuning with trainable addresses

In code, the main variant is `learned_address_k32` / `LearnedAddressSparseAttention` (`k32` = address dimension $d_A{=}32$, not forward $K$).

## Main results (canonical proof cell)

| Variant | NIAH accuracy @ $T{=}2048$ | Notes |
|---------|------------------------------|-------|
| Dense Flash | 100.0% | Teacher |
| **Wayfinder** (`learned_address_k32`) | **100.0%** | Matches dense through $T{=}4096$ |
| Key-vector top-$K$ | 81.3% | Same forward $K{=}128$, different geometry |
| Local window $W{=}64$ | 12.3% | Fixed-pattern sparse baseline |

At $T{=}16384$, a fused forward prototype reaches **~2.2× lower latency** and **~2.4× less peak VRAM** than dense FlashAttention (see `systems_benchmark.json`).

This repository is intentionally narrow: a reproducible dense-to-sparse recipe, matched-budget ablations, and a systems prototype — not a frontier LLM training stack.

## Repository layout

```
routing_attention/          Core library (attention, retrieval kernels, benchmarks)
configs/                    Experiment YAML (canonical: learned_address_proof_cell/)
scripts/                    Verification, figure generation, diagnostics
run_*.py                    Suite orchestrators (proof cell is the paper entry point)
experiments/
  experiment_*.py           Legacy experiment entrypoints (code only)
  Experiment_7/
    learned_address_proof_cell/proof_cell_full/   Canonical JSON results (tracked)
    learned_address_breakthrough/results_table_full.json
paper_figures/              Paper figures (regenerable)
wayfinder_attention_paper_v0_4.tex
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (experiments were run on RTX 5090 32GB under WSL)
- PyTorch ≥ 2.4 with FlashAttention-compatible setup

```bash
pip install -r requirements.txt
```

MNIST (used in some auxiliary experiments) is downloaded automatically by torchvision into `data/` on first use — this directory is not tracked in git.

## Quick start

### 1. Verify the canonical config

Smoke-check that the proof-cell YAML, task generator, and holdout protocol load correctly (no GPU training):

```bash
python scripts/verify_learned_address_proof_cell.py
```

### 2. Regenerate paper figures

Figures are built from tracked canonical JSON artifacts (no retraining):

```bash
python scripts/generate_wayfinder_paper_figures.py
```

Outputs land in `paper_figures/` (`fig_architecture.png`, `fig_protocol.png`, `fig_depth_bars.png`, `fig_latency.png`, `fig_bsweep.png`, `fig_retrieval_tiling.png`).

### 3. Inspect canonical results

| Artifact | Path |
|----------|------|
| Per-variant accuracies, recall, depth breakdown | `experiments/Experiment_7/learned_address_proof_cell/proof_cell_full/` |
| Aggregated results table (figures) | `experiments/Experiment_7/learned_address_breakthrough/results_table_full.json` |
| Systems latency / VRAM benchmark | `experiments/Experiment_7/learned_address_proof_cell/proof_cell_full/systems_benchmark.json` |

## Reproducing the proof cell

Full training retrains Phase A → B → C and writes fresh checkpoints under `experiments/Experiment_7/learned_address_proof_cell/proof_cell_full/`. **Always dry-run first** to validate the pipeline on your machine:

```bash
# Smoke test (~minutes): short step budgets, validates wiring
python run_learned_address_proof_cell_suite.py --phase proof_cell --dry-run

# Full canonical proof cell (hours on a single GPU)
python run_learned_address_proof_cell_suite.py --phase proof_cell
```

Optional extended phases from the paper:

```bash
python run_learned_address_proof_cell_suite.py --phase sweep      # Phase B step sweep
python run_learned_address_proof_cell_suite.py --phase curriculum # T=2048→4096→8192
python run_learned_address_proof_cell_suite.py --phase all        # all phases
```

Canonical config: `configs/learned_address_proof_cell/niah_pointer_unique_t2048_4L.yaml`  
(4-layer model, $T{=}2048$, seed 45, `pointer_unique`, 0 decoys, 300-example holdout)

## What is tracked in git

This repo follows a **code + summary JSON** release model (~3–4 MB tracked):

| Tracked | Not tracked (see `.gitignore`) |
|---------|-------------------------------|
| Source, configs, suite scripts | Model checkpoints (`*.pt`) |
| Canonical proof-cell JSON summaries | Attention caches, tensorboard, plots |
| Paper `.tex` and `paper_figures/*.png` | Numbered run dirs (`run_001` … `run_N`) |
| `requirements.txt` | Suite output trees, dry-run artifacts, logs |

Checkpoints and full training artifacts can be obtained by re-running the suite locally or from the Zenodo publication bundle.

## Other suite scripts

The repository includes many `run_*_suite.py` orchestrators from the broader research program (routing arena, gap calibration, slot-pointer ablations, etc.). These are preserved for transparency but are **not** required to reproduce the main paper tables; start with `run_learned_address_proof_cell_suite.py`.

## Citation

If you use this code or build on these results, please cite the Wayfinder Attention paper:

```bibtex
@article{anikevich2026wayfinder,
  title   = {Wayfinder Attention: Attention as Learned Search},
  author  = {Anikevich, Aleksandr},
  year    = {2026},
  doi     = {10.5281/zenodo.21087516},
  url     = {https://doi.org/10.5281/zenodo.21087516}
}
```

## Contact

Aleksandr Anikevich — [aleksandranikevich2@gmail.com](mailto:aleksandranikevich2@gmail.com)

## License

See the Zenodo record for the paper license. Add a `LICENSE` file here if you choose a specific open-source license for the code (MIT/Apache-2.0 are common choices for research code releases).
