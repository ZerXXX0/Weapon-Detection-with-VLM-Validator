#!/usr/bin/env python3
"""
Batch Training Orchestrator for 10 VLM Configurations (Experiments 2 & 3)
5 Models x 2 Context Modes = 10 Experiments:
1.  qwen2_vl_2b_single  (Qwen2-VL-2B, Full Image Only)
2.  qwen2_vl_2b_dual    (Qwen2-VL-2B, Full Image + 20% Zoomed-Out Crop)
3.  qwen2_vl_7b_single  (Qwen2-VL-7B, Full Image Only)
4.  qwen2_vl_7b_dual    (Qwen2-VL-7B, Full Image + 20% Zoomed-Out Crop)
5.  qwen3_vl_2b_single  (Qwen3-VL-2B, Full Image Only)
6.  qwen3_vl_2b_dual    (Qwen3-VL-2B, Full Image + 20% Zoomed-Out Crop)
7.  qwen3_vl_4b_single  (Qwen3-VL-4B, Full Image Only)
8.  qwen3_vl_4b_dual    (Qwen3-VL-4B, Full Image + 20% Zoomed-Out Crop)
9.  qwen3_vl_8b_single  (Qwen3-VL-8B, Full Image Only)
10. qwen3_vl_8b_dual    (Qwen3-VL-8B, Full Image + 20% Zoomed-Out Crop)

Reference: writing-block.md (Sections 4 & 5)
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

ALL_MODELS = ["qwen2-2b", "qwen2-7b", "qwen3-2b", "qwen3-4b", "qwen3-8b"]
ALL_CONTEXTS = ["single", "dual"]

MODEL_PRETTY = {
    "qwen2-2b": "Qwen2-VL-2B",
    "qwen2-7b": "Qwen2-VL-7B",
    "qwen3-2b": "Qwen3-VL-2B",
    "qwen3-4b": "Qwen3-VL-4B",
    "qwen3-8b": "Qwen3-VL-8B",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator for VLM 10-Model Benchmarking Experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Models to train: {ALL_MODELS} or 'all'",
    )
    parser.add_argument(
        "--context-modes",
        nargs="+",
        default=["both"],
        choices=["single", "dual", "both"],
        help="Context modes: 'single', 'dual', or 'both'",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="2-class-weapon-dataset-3", help="Base dataset directory for images")
    parser.add_argument("--train-json", type=str, default=None, help="Custom train.json path")
    parser.add_argument("--valid-json", type=str, default=None, help="Custom valid.json path")
    parser.add_argument("--test-json", type=str, default=None, help="Custom test.json path")
    parser.add_argument("--output-style", type=str, default="cot", choices=["cot", "binary"], help="cot (Reasoning + Decision) or binary (WEAPON/NOT_WEAPON)")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip if adapter checkpoint already exists")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Display the execution schedule without training")
    return parser.parse_args()

def is_already_trained(adapter_dir):
    p = Path(adapter_dir)
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )

def main():
    args = parse_args()

    # Determine target models
    if "all" in args.models:
        target_models = ALL_MODELS
    else:
        target_models = [m for m in args.models if m in ALL_MODELS]

    # Determine target context modes
    if "both" in args.context_modes:
        target_contexts = ALL_CONTEXTS
    else:
        target_contexts = [c for c in args.context_modes if c in ALL_CONTEXTS]

    # Generate experiment schedule
    experiments = []
    for model_key in target_models:
        for ctx in target_contexts:
            exp_id = f"{model_key.replace('-', '_')}_{ctx}"
            adapter_dir = ROOT / f"adapters/vlm/{exp_id}"
            experiments.append({
                "exp_id": exp_id,
                "model_key": model_key,
                "model_name": MODEL_PRETTY[model_key],
                "context_mode": ctx,
                "adapter_dir": str(adapter_dir),
            })

    print("=" * 75)
    print("VLM 10-EXPERIMENT TRAINING SCHEDULE (Experiments 2 & 3)")
    print("=" * 75)
    print(f"Total Configurations Scheduled: {len(experiments)}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}) | LR: {args.learning_rate}")
    print("-" * 75)
    for idx, exp in enumerate(experiments, 1):
        status = "[DONE]" if is_already_trained(exp["adapter_dir"]) else "[PENDING]"
        ctx_desc = "Single-Image (Full Image Only)" if exp["context_mode"] == "single" else "Dual-Image (Full Image + 20% Crop)"
        print(f" {idx:2d}. {exp['model_name']:<12} | Context: {ctx_desc:<35} | {status} -> {exp['adapter_dir']}")
    print("=" * 75)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No models were trained.")
        return

    # Execute training sequence
    for idx, exp in enumerate(experiments, 1):
        exp_id = exp["exp_id"]
        adapter_dir = exp["adapter_dir"]

        if args.skip_existing and is_already_trained(adapter_dir):
            print(f"\n>>> Skipping [{idx}/{len(experiments)}] {exp_id} (Checkpoint exists in {adapter_dir})")
            continue

        print(f"\n{'#' * 75}")
        print(f">>> STARTING [{idx}/{len(experiments)}]: {exp['model_name']} ({exp['context_mode'].upper()})")
        print(f"{'#' * 75}\n")

        train_script = str(ROOT / "experiments/vlm_benchmark/train_vlm.py")
        cmd = [
            PYTHON,
            train_script,
            "--model-key", exp["model_key"],
            "--context-mode", exp["context_mode"],
            "--output-dir", adapter_dir,
            "--data-dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--learning-rate", str(args.learning_rate),
            "--output-style", args.output_style,
        ]
        if args.train_json:
            cmd.extend(["--train-json", args.train_json])
        if args.valid_json:
            cmd.extend(["--valid-json", args.valid_json])
        if args.test_json:
            cmd.extend(["--test-json", args.test_json])

        t_start = time.time()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        elapsed_min = (time.time() - t_start) / 60.0

        if ret.returncode != 0:
            print(f"\n[ERROR] Experiment {exp_id} failed with exit code {ret.returncode}!")
            sys.exit(ret.returncode)
        else:
            print(f"\n[SUCCESS] Completed {exp_id} in {elapsed_min:.2f} minutes.")

    print("\n" + "=" * 75)
    print("ALL SCHEDULED VLM EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
