#!/usr/bin/env python3
"""
Batch Evaluation Orchestrator for All 10 Trained VLM Adapters (Experiments 2 & 3)
Evaluates all adapters in adapters/vlm/ sequentially on the test set.

Reference: writing-block.md (Sections 4 & 5)
"""

import os
import sys
import glob
import subprocess
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

ALL_ADAPTERS = [
    "qwen2_2b_single",
    "qwen2_2b_dual",
    "qwen2_7b_single",
    "qwen2_7b_dual",
    "qwen3_2b_single",
    "qwen3_2b_dual",
    "qwen3_4b_single",
    "qwen3_4b_dual",
    "qwen3_8b_single",
    "qwen3_8b_dual",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate All Trained VLM Adapters")
    parser.add_argument("--adapters", nargs="+", default=["all"], help="Specific adapter names or 'all'")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of test samples per model (default 200, 0 for all)")
    parser.add_argument("--csv-output", type=str, default="results/metrics/vlm_benchmark_real.csv", help="Consolidated CSV output path")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print plan without running")
    return parser.parse_args()

def main():
    args = parse_args()
    os.chdir(ROOT)

    if "all" in args.adapters:
        targets = ALL_ADAPTERS
    else:
        targets = args.adapters

    available = []
    for a in targets:
        p = ROOT / f"adapters/vlm/{a}"
        if p.is_dir() and ((p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file()):
            available.append(a)
        else:
            print(f"[SKIP] Adapter {a} not found or missing weights in adapters/vlm/{a}")

    print("=" * 70)
    print("VLM TEST SET EVALUATION BATCH (Experiments 2 & 3)")
    print("=" * 70)
    print(f"Total Available Adapters to Evaluate: {len(available)}")
    print(f"Samples per Model: {args.num_samples if args.num_samples > 0 else 'Full Test Set'}")
    print(f"Output CSV: {args.csv_output}")
    print("-" * 70)
    for idx, a in enumerate(available, 1):
        print(f"  {idx:2d}. {a}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] No evaluations executed.")
        return

    # Delete existing CSV if starting fresh to avoid duplicate rows
    out_csv = ROOT / args.csv_output
    if out_csv.exists():
        backup = out_csv.with_suffix(f".bak_{int(time.time())}.csv")
        out_csv.rename(backup)
        print(f"Existing {out_csv.name} backed up to {backup.name}")

    eval_script = str(ROOT / "experiments/vlm_benchmark/eval_vlm.py")

    for idx, a in enumerate(available, 1):
        print(f"\n{'#' * 70}")
        print(f">>> EVALUATING [{idx}/{len(available)}]: {a}")
        print(f"{'#' * 70}\n")

        adapter_dir = str(ROOT / f"adapters/vlm/{a}")
        cmd = [
            PYTHON,
            eval_script,
            "--adapter-dir", adapter_dir,
            "--num-samples", str(args.num_samples),
            "--csv-output", str(out_csv),
        ]

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        el_min = (time.time() - t0) / 60.0

        if ret.returncode != 0:
            print(f"[ERROR] Evaluation of {a} failed with exit code {ret.returncode}!")
        else:
            print(f"[SUCCESS] Finished {a} in {el_min:.2f} min.")

    print("\n" + "=" * 70)
    print(f"ALL VLM EVALUATIONS COMPLETE! Results saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
