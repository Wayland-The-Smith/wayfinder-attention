#!/usr/bin/env python3
"""Evaluate one or more attention variants on the long-context benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.experiment_7 import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="routing_asymmetric")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--context-lens", type=int, nargs="*")
    args = parser.parse_args()

    override = {}
    if args.context_lens:
        override["long_context_benchmark"] = {"context_lengths": args.context_lens}

    run(
        variant=args.variant,
        dry_run=args.dry_run,
        config_override=override or None,
        skip_training=args.skip_training,
        eval_only_context_lengths=args.context_lens,
    )


if __name__ == "__main__":
    main()
