#!/usr/bin/env python3
import os
import sys
import time
import glob
import json
import csv
import argparse
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.common.dual_image_utils import load_yolo_ground_truth, prepare_dual_images, format_coordinate_prompt
from experiments.common.metrics import calculate_iou

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone VLM Benchmark (Experiment 2)")
    parser.add_argument("--model-name", type=str, default="Qwen3-VL-8B", help="Model label")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--adapter-path", type=str, default="qwen3_vl_model_2image", help="PEFT adapter path if any")
    parser.add_argument("--dataset-dir", type=str, default="1-class-weapon-dataset-2", help="Dataset directory")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of balanced test evaluation samples (half weapon, half non-weapon)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-4bit", action="store_true", default=True, help="Load in 4-bit NF4 quantization")
    parser.add_argument("--csv-output", type=str, default="results/metrics/vlm_benchmark.csv")
    parser.add_argument("--raw-log", type=str, default=None)
    return parser.parse_args()

def extract_balanced_evaluation_samples(dataset_dir, num_samples=200, seed=42):
    """
    Extracts balanced ground-truth evaluation samples:
    - 50% Ground Truth Weapons (Positive)
    - 50% Verified Non-Weapon Background Crops (Negative with IoU=0 against all GTs)
    """
    random.seed(seed)
    np.random.seed(seed)

    test_img_dir = os.path.join(dataset_dir, "test", "images")
    test_lbl_dir = os.path.join(dataset_dir, "test", "labels")

    image_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png")))

    positives = []
    negatives = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        with Image.open(img_path) as im:
            w, h = im.size

        lbl_file = os.path.splitext(filename)[0] + ".txt"
        lbl_path = os.path.join(test_lbl_dir, lbl_file)
        gt_boxes, gt_labels = load_yolo_ground_truth(lbl_path, w, h)

        # Collect ground truth weapons
        for b in gt_boxes:
            positives.append({
                "image_path": img_path,
                "box": b.tolist(),
                "label": "WEAPON",
                "binary_target": 1
            })

        # Sample non-overlapping background negative box
        if len(gt_boxes) > 0:
            for _ in range(3):
                bw = random.uniform(0.1, 0.4) * w
                bh = random.uniform(0.1, 0.4) * h
                bx1 = random.uniform(0, w - bw)
                by1 = random.uniform(0, h - bh)
                cand_box = [bx1, by1, bx1 + bw, by1 + bh]

                max_iou = max([calculate_iou(cand_box, gtb) for gtb in gt_boxes]) if len(gt_boxes) > 0 else 0.0
                if max_iou == 0.0:
                    negatives.append({
                        "image_path": img_path,
                        "box": cand_box,
                        "label": "NOT_WEAPON",
                        "binary_target": 0
                    })
                    break

    target_per_class = num_samples // 2
    random.shuffle(positives)
    random.shuffle(negatives)

    selected_pos = positives[:target_per_class]
    selected_neg = negatives[:target_per_class]

    combined = selected_pos + selected_neg
    random.shuffle(combined)
    return combined

def normalize_vlm_response(raw_text):
    """
    Standardize model response to 'WEAPON', 'NOT_WEAPON', or 'INVALID'.
    """
    cleaned = raw_text.strip().upper()
    if "NOT_WEAPON" in cleaned or "NON-WEAPON" in cleaned or "NOT WEAPON" in cleaned or "FALSE ALARM" in cleaned:
        return "NOT_WEAPON"
    elif "WEAPON" in cleaned or "IDENTIFIED AS A WEAPON" in cleaned or "YES" in cleaned:
        return "WEAPON"
    else:
        return "INVALID"

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    os.makedirs("results/raw_predictions", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    if args.raw_log is None:
        args.raw_log = f"results/raw_predictions/vlm_standalone_{args.model_name.lower().replace('-', '_')}.jsonl"

    print("=" * 60)
    print(f"Standalone VLM Benchmark: {args.model_name}")
    print(f"Model ID: {args.model_id} | Adapter: {args.adapter_path}")
    print("=" * 60)

    # 1. Prepare evaluation samples
    samples = extract_balanced_evaluation_samples(args.dataset_dir, num_samples=args.num_samples)
    print(f"Extracted {len(samples)} balanced test samples ({sum(s['binary_target'] for s in samples)} Weapon, {sum(1 - s['binary_target'] for s in samples)} Non-Weapon)")

    # 2. Load VLM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_load_start = time.time()
    from transformers import AutoProcessor

    bnb_config = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    # Model loader based on model type
    if "Qwen3-VL" in args.model_id:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if args.device == "cuda" else None,
            torch_dtype=torch.float16
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if args.device == "cuda" else None,
            torch_dtype=torch.float16
        )

    if args.adapter_path and os.path.exists(args.adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.adapter_path if (args.adapter_path and os.path.exists(args.adapter_path)) else args.model_id,
        min_pixels=256*256,
        max_pixels=768*768
    )

    load_time_sec = round(time.time() - t_load_start, 2)
    param_count = sum(p.numel() for p in model.parameters())

    # 3. Evaluation Loop
    results_log = []
    latencies = []
    y_true = []
    y_pred = []
    ambiguous_count = 0

    for sample in tqdm(samples, desc=f"Evaluating {args.model_name}"):
        img_path = sample["image_path"]
        box = sample["box"]
        target = sample["label"]

        image_pil = Image.open(img_path).convert("RGB")
        w, h = image_pil.size

        # Use Dual-Image context
        img1_full, img2_crop = prepare_dual_images(image_pil, box)

        prompt_text = (
            "<|im_start|>system\nYou are a precise weapon detection verification engine.<|im_end|>\n"
            "<|im_start|>user\n"
            "Examine the candidate object in the images. Determine if the highlighted region contains a real weapon (handgun, firearm, rifle, knife) or a false alarm/everyday object.\n"
            "Respond strictly with either 'WEAPON' or 'NOT_WEAPON'.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1_full},
                    {"type": "image", "image": img2_crop},
                    {"type": "text", "text": "Is the candidate object a weapon or not a weapon? Respond with 'WEAPON' or 'NOT_WEAPON'."}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[img1_full, img2_crop],
            padding=True,
            return_tensors="pt"
        ).to(args.device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

        in_ids = inputs["input_ids"]
        out_text = processor.batch_decode(
            generated_ids[:, in_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        pred_normalized = normalize_vlm_response(out_text)
        if pred_normalized == "INVALID":
            ambiguous_count += 1
            pred_binary = 0  # Default fallback for ambiguity
        else:
            pred_binary = 1 if pred_normalized == "WEAPON" else 0

        y_true.append(sample["binary_target"])
        y_pred.append(pred_binary)

        sample_record = {
            "image": os.path.basename(img_path),
            "box": box,
            "target": target,
            "target_binary": sample["binary_target"],
            "raw_output": out_text.strip(),
            "predicted": pred_normalized,
            "predicted_binary": pred_binary,
            "latency_ms": round((t1 - t0) * 1000.0, 2)
        }
        results_log.append(sample_record)

    # Save raw log
    with open(args.raw_log, "w", encoding="utf-8") as f:
        for r in results_log:
            f.write(json.dumps(r) + "\n")

    # 4. Metrics computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Rejection rate: true negative rate (how many false alarms were successfully rejected)
    fprr = tn / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0
    # FN introduction: false negative rate
    fnir = fn / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    peak_gpu_mem = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    metrics_row = {
        "model_name": args.model_name,
        "model_id": args.model_id,
        "parameters": param_count,
        "load_time_sec": load_time_sec,
        "total_samples": len(y_true),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "ambiguous_responses": ambiguous_count,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "fprr_percent": round(fprr, 2),
        "fnir_percent": round(fnir, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_img_per_sec": round(throughput, 2),
        "peak_gpu_memory_mb": peak_gpu_mem
    }

    # Write to CSV
    file_exists = os.path.exists(args.csv_output)
    with open(args.csv_output, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)

    print("\n" + "=" * 60)
    print("Standalone VLM Benchmark Results:")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"FPRR (False Alarm Rejection): {fprr:.2f}% | FNIR: {fnir:.2f}%")
    print(f"Average Latency: {avg_lat:.2f} ms | P95 Latency: {p95_lat:.2f} ms | Peak VRAM: {peak_gpu_mem} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
