#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()
    
    print("=" * 80)
    print(" STEP 1: Starting Qwen3-8B 2-Image Fine-Tuning")
    print("=" * 80)
    
    train_script = script_dir / "train_qwen3_2image.py"
    train_cmd = [
        sys.executable, str(train_script),
        "--output-dir", "./qwen3_vl_model_2image"
    ]
    ret_train = subprocess.run(train_cmd)
    if ret_train.returncode != 0:
        print("[ERROR] Training failed. Aborting evaluation.")
        sys.exit(ret_train.returncode)

    print("\n" + "=" * 80)
    print(" STEP 2: Starting YOLO12 + Dual-Image VLM Evaluation")
    print("=" * 80)

    eval_script = script_dir / "validate_with_vlm_qwen3_2image_yolo12.py"
    eval_cmd = [
        sys.executable, str(eval_script),
        "--qwen-model", "./qwen3_vl_model_2image"
    ]
    ret_eval = subprocess.run(eval_cmd)
    sys.exit(ret_eval.returncode)

if __name__ == "__main__":
    main()
