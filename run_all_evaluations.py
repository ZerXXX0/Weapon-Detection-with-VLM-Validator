#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def run_evaluation(script_path, description):
    print("\n" + "=" * 80)
    print(f" LAUNCHING EVALUATION: {description}")
    print(f" Script: {script_path}")
    print("=" * 80 + "\n")
    
    cmd = [sys.executable, str(script_path)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[WARNING] Evaluation failed for {script_path} with exit code {ret.returncode}")
    else:
        print(f"[SUCCESS] Completed evaluation for {description}")

def main():
    root_dir = Path(__file__).parent.resolve()
    
    evaluations = [
        (
            root_dir / "dual_image_qwen3_8b_yolo12" / "validate_with_vlm_qwen3_2image_yolo12.py",
            "YOLO12 + Dual-Image Qwen3-VL (20% Zoomed-Out Crop)"
        ),
        (
            root_dir / "dual_image_qwen3_8b" / "validate_with_vlm_qwen3_2image_yolo26.py",
            "YOLO26 + Dual-Image Qwen3-VL (20% Zoomed-Out Crop)"
        ),
        (
            root_dir / "validate_with_vlm_qwen3_yolo26.py",
            "YOLO26 + Single-Image Qwen3-VL"
        ),
        (
            root_dir / "validate_with_vlm_qwen3.py",
            "YOLO12 + Single-Image Qwen3-VL"
        ),
    ]

    for script_path, desc in evaluations:
        if script_path.exists():
            run_evaluation(script_path, desc)
        else:
            print(f"[ERROR] Could not find script at: {script_path}")

    print("\n" + "=" * 80)
    print(" ALL 4 EVALUATIONS COMPLETED!")
    print("=========================================================")

if __name__ == "__main__":
    main()
