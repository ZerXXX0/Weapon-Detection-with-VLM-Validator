#!/usr/bin/env python3
import subprocess
import sys

def main():
    print("=" * 80)
    print(" STEP 1: Starting Qwen3-8B 2-Image Fine-Tuning")
    print("=" * 80)
    
    train_cmd = [
        sys.executable, "dual_image_qwen3_8b/train_qwen3_2image.py",
        "--output-dir", "./qwen3_vl_model_2image"
    ]
    ret_train = subprocess.run(train_cmd)
    if ret_train.returncode != 0:
        print("[ERROR] Training failed. Aborting evaluation.")
        sys.exit(ret_train.returncode)

    print("\n" + "=" * 80)
    print(" STEP 2: Starting YOLO + Dual-Image VLM Evaluation")
    print("=" * 80)

    eval_cmd = [
        sys.executable, "dual_image_qwen3_8b/validate_with_vlm_qwen3_2image_yolo26.py",
        "--qwen-model", "./qwen3_vl_model_2image"
    ]
    ret_eval = subprocess.run(eval_cmd)
    sys.exit(ret_eval.returncode)

if __name__ == "__main__":
    main()
