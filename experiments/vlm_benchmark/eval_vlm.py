#!/usr/bin/env python3
"""
Standalone VLM Evaluation on Test Set (Experiments 2 & 3)
Evaluates a fine-tuned LoRA adapter on ground truth weapon & non-weapon test samples.
Supports both Single-Image (Full Image Only) and Dual-Image (Full Image + 20% Crop).

Reference: writing-block.md (Sections 4 & 5)
"""

import os
import sys
import re
import time
import json
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

# Default annotation and image search paths
def find_default_test_json():
    possible = [
        ROOT / "Dataset Annotation-20260807T120404Z-1-001-20261117T042214Z-1-001/Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation/test.json",
        ROOT / "test.json",
    ]
    for p in possible:
        if p.is_file():
            return str(p)
    return "test.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM Adapter on Test Set")
    parser.add_argument("--adapter-dir", type=str, required=True, help="Path to trained adapter directory (e.g. adapters/vlm/qwen3_8b_dual)")
    parser.add_argument("--test-json", type=str, default=find_default_test_json(), help="Path to test.json annotations")
    parser.add_argument("--data-dir", type=str, default="2-class-weapon-dataset-3", help="Base dataset directory containing images")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of balanced test evaluation samples (default 200, 0 for all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-4bit", action="store_true", default=True, help="Load in 4-bit NF4 quantization")
    parser.add_argument("--margin-ratio", type=float, default=0.20, help="Zoom-out margin for cropped context in dual mode")
    parser.add_argument("--csv-output", type=str, default="results/metrics/vlm_benchmark_real.csv", help="Output metrics CSV")
    parser.add_argument("--raw-log", type=str, default=None, help="Raw per-sample JSONL output log")
    return parser.parse_args()

def resolve_image_path(image_path, fallback_dir="2-class-weapon-dataset-3"):
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate

    stripped = Path(*candidate.parts[1:]) if (candidate.parts and candidate.parts[0] == "dataset") else candidate

    search_dirs = [
        Path(fallback_dir),
        Path("2-class-weapon-dataset-3"),
        Path("1-class-weapon-dataset-2"),
        Path("."),
    ]

    for s_dir in search_dirs:
        cand1 = s_dir / stripped
        if cand1.is_file():
            return cand1
        cand2 = s_dir / candidate
        if cand2.is_file():
            return cand2
        for split in ("test", "valid", "train"):
            split_cand = s_dir / split / "images" / candidate.name
            if split_cand.is_file():
                return split_cand
        b_name = s_dir / candidate.name
        if b_name.is_file():
            return b_name

    raise FileNotFoundError(f"Could not resolve image path: {image_path}")

def parse_box_coords(text_content):
    match = re.search(r"<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>", text_content)
    if match:
        return tuple(map(int, match.groups()))
    return None

def create_visual_inputs(image_path, box_coords, context_mode="dual", margin_ratio=0.20):
    img_full = Image.open(image_path).convert("RGB")
    width, height = img_full.size

    if context_mode == "single":
        return [img_full]

    img1 = img_full.copy()
    draw = ImageDraw.Draw(img1)

    if box_coords:
        ymin_1000, xmin_1000, ymax_1000, xmax_1000 = box_coords
        xmin_px = (xmin_1000 / 1000.0) * width
        ymin_px = (ymin_1000 / 1000.0) * height
        xmax_px = (xmax_1000 / 1000.0) * width
        ymax_px = (ymax_1000 / 1000.0) * height

        for offset in range(3):
            draw.rectangle(
                [xmin_px - offset, ymin_px - offset, xmax_px + offset, ymax_px + offset],
                outline=(255, 0, 0),
            )

        bw = xmax_px - xmin_px
        bh = ymax_px - ymin_px
        pad_w = bw * margin_ratio
        pad_h = bh * margin_ratio

        crop_xmin = max(0, int(round(xmin_px - pad_w)))
        crop_ymin = max(0, int(round(ymin_px - pad_h)))
        crop_xmax = min(width, int(round(xmax_px + pad_w)))
        crop_ymax = min(height, int(round(ymax_px + pad_h)))

        if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
            img2 = img_full.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        else:
            img2 = img_full.copy()
    else:
        img2 = img_full.copy()

    return [img1, img2]

def normalize_vlm_response(raw_text):
    """
    Standardize model response to 'WEAPON', 'NOT_WEAPON', or 'INVALID'.
    Handles both CoT ('Final Decision: Weapon/Non-Weapon') and direct binary output.
    """
    cleaned = raw_text.strip().upper()
    if "FINAL DECISION: NON-WEAPON" in cleaned or "NON-WEAPON" in cleaned or "NOT_WEAPON" in cleaned or "NOT WEAPON" in cleaned or "FALSE ALARM" in cleaned:
        return "NOT_WEAPON"
    elif "FINAL DECISION: WEAPON" in cleaned or "WEAPON" in cleaned or "FIREARM" in cleaned:
        return "WEAPON"
    else:
        return "INVALID"

def load_evaluation_samples(test_json, data_dir, num_samples=200, seed=42):
    """
    Loads samples from test_json. If num_samples > 0, extracts a balanced subset (50% weapon, 50% non-weapon).
    """
    import random
    random.seed(seed)

    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = []
    negatives = []

    for item in data:
        user_msg = item["messages"][0]["content"]
        asst_msg = item["messages"][1]["content"][0]["text"]

        raw_img = user_msg[0]["image"]
        raw_txt = user_msg[1]["text"]

        upper_asst = asst_msg.upper()
        if "NON-WEAPON" in upper_asst or "NOT_WEAPON" in upper_asst or "NOT WEAPON" in upper_asst or "FALSE ALARM" in upper_asst:
            target_label = "NOT_WEAPON"
            binary_target = 0
            negatives.append({
                "image_raw": raw_img,
                "text": raw_txt,
                "label": target_label,
                "binary_target": binary_target,
                "gt_reasoning": asst_msg,
            })
        else:
            target_label = "WEAPON"
            binary_target = 1
            positives.append({
                "image_raw": raw_img,
                "text": raw_txt,
                "label": target_label,
                "binary_target": binary_target,
                "gt_reasoning": asst_msg,
            })

    if num_samples <= 0 or num_samples >= len(data):
        all_samples = positives + negatives
        random.shuffle(all_samples)
        return all_samples

    half = num_samples // 2
    n_pos = min(half, len(positives))
    n_neg = min(half, len(negatives))

    # If negatives are fewer than half, take all negatives and fill the rest with positives
    if len(negatives) < half:
        n_neg = len(negatives)
        n_pos = num_samples - n_neg

    random.shuffle(positives)
    random.shuffle(negatives)

    selected = positives[:n_pos] + negatives[:n_neg]
    random.shuffle(selected)
    return selected

def main():
    args = parse_args()
    adapter_path = Path(args.adapter_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    # Read metadata if present
    meta_path = adapter_path / "experiment_metadata.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model_name = metadata.get("model_name", adapter_path.name)
        model_id = metadata.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        model_class_name = metadata.get("model_class", "Qwen3VLForConditionalGeneration")
        context_mode = metadata.get("context_mode", "dual" if "dual" in adapter_path.name else "single")
    else:
        # Infer from folder name
        name_lower = adapter_path.name.lower()
        context_mode = "dual" if "dual" in name_lower else "single"
        if "qwen3_8b" in name_lower:
            model_name = "Qwen3-VL-8B"; model_id = "Qwen/Qwen3-VL-8B-Instruct"; model_class_name = "Qwen3VLForConditionalGeneration"
        elif "qwen3_4b" in name_lower:
            model_name = "Qwen3-VL-4B"; model_id = "Qwen/Qwen3-VL-4B-Instruct"; model_class_name = "Qwen3VLForConditionalGeneration"
        elif "qwen3_2b" in name_lower:
            model_name = "Qwen3-VL-2B"; model_id = "Qwen/Qwen3-VL-2B-Instruct"; model_class_name = "Qwen3VLForConditionalGeneration"
        elif "qwen2_7b" in name_lower:
            model_name = "Qwen2-VL-7B"; model_id = "Qwen/Qwen2-VL-7B-Instruct"; model_class_name = "Qwen2VLForConditionalGeneration"
        else:
            model_name = "Qwen2-VL-2B"; model_id = "Qwen/Qwen2-VL-2B-Instruct"; model_class_name = "Qwen2VLForConditionalGeneration"

    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    os.makedirs("results/raw_predictions", exist_ok=True)

    if args.raw_log is None:
        args.raw_log = f"results/raw_predictions/eval_{adapter_path.name}.jsonl"

    print("=" * 65)
    print(f"EVALUATING VLM ADAPTER: {adapter_path.name}")
    print(f"Base Model    : {model_name} ({model_id})")
    print(f"Context Mode  : {context_mode.upper()} ({'Full Image Only' if context_mode == 'single' else 'Full Image + 20% Crop'})")
    print(f"Test Dataset  : {args.test_json}")
    print(f"Adapter Path  : {adapter_path}")
    print("=" * 65)

    # 1. Load samples
    samples = load_evaluation_samples(args.test_json, args.data_dir, num_samples=args.num_samples)
    n_weapons = sum(s["binary_target"] for s in samples)
    n_non_weapons = len(samples) - n_weapons
    print(f"Loaded {len(samples)} evaluation samples ({n_weapons} Weapon, {n_non_weapons} Non-Weapon)")

    # 2. Load model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_load = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_4bit else None

    if model_class_name == "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if args.device == "cuda" else None,
            torch_dtype=torch.float16,
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if args.device == "cuda" else None,
            torch_dtype=torch.float16,
        )

    from peft import PeftModel
    print(f"Loading LoRA weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    processor = AutoProcessor.from_pretrained(
        str(adapter_path) if (adapter_path / "processor_config.json").exists() else model_id,
        min_pixels=256 * 256,
        max_pixels=768 * 768,
    )

    load_time_sec = round(time.time() - t_load, 2)
    param_count = sum(p.numel() for p in model.parameters())

    # 3. Evaluation loop
    results_log = []
    latencies = []
    y_true = []
    y_pred = []
    ambiguous_count = 0

    system_instruction = "<|im_start|>system\nYou are a precise weapon detection verification engine.<|im_end|>\n"

    for sample in tqdm(samples, desc=f"Evaluating {adapter_path.name}"):
        raw_img_path = sample["image_raw"]
        raw_text = sample["text"]
        target = sample["label"]

        resolved_img = resolve_image_path(raw_img_path, args.data_dir)
        box_coords = parse_box_coords(raw_text)

        images = create_visual_inputs(resolved_img, box_coords, context_mode=context_mode, margin_ratio=args.margin_ratio)

        if context_mode == "dual":
            user_text = (
                "Picture 1 shows the full image with the candidate object highlighted in a bounding box. "
                "Picture 2 shows a cropped, 20% zoomed-out view of the object for detailed inspection. "
                "Analyze the object located inside the bounding box and determine whether the object is a real weapon or a visually similar non-weapon object."
            )
            image_tokens = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
        else:
            coord_str = f"<box>{box_coords}</box>" if box_coords else "the highlighted coordinates"
            user_text = (
                f"Picture 1 shows the full scene. The detected candidate object is at {coord_str}. "
                "Analyze the object located inside the bounding box and determine whether the object is a real weapon or a visually similar non-weapon object."
            )
            image_tokens = "<|vision_start|><|image_pad|><|vision_end|>"

        full_prompt = f"{system_instruction}<|im_start|>user\n{image_tokens}{user_text}\n{raw_text}<|im_end|>\n<|im_start|>assistant\n"

        inputs = processor(
            text=[full_prompt],
            images=[images] if len(images) > 1 else images,
            padding=True,
            return_tensors="pt",
        ).to(args.device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        t1 = time.time()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)

        in_ids = inputs["input_ids"]
        out_text = processor.batch_decode(
            generated_ids[:, in_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        pred_normalized = normalize_vlm_response(out_text)
        if pred_normalized == "INVALID":
            ambiguous_count += 1
            pred_binary = 0
        else:
            pred_binary = 1 if pred_normalized == "WEAPON" else 0

        y_true.append(sample["binary_target"])
        y_pred.append(pred_binary)

        results_log.append({
            "image": os.path.basename(resolved_img),
            "target": target,
            "target_binary": sample["binary_target"],
            "raw_output": out_text.strip(),
            "predicted": pred_normalized,
            "predicted_binary": pred_binary,
            "latency_ms": round(lat_ms, 2),
        })

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
    fprr = tn / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0
    fnir = fn / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    peak_gpu_mem = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    metrics_row = {
        "adapter_name": adapter_path.name,
        "model_name": model_name,
        "model_id": model_id,
        "context_mode": context_mode,
        "parameters": param_count,
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
        "fprr_percent": round(fprr, 2),
        "fnir_percent": round(fnir, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_crops_per_sec": round(throughput, 2),
        "peak_gpu_memory_mb": peak_gpu_mem,
    }

    file_exists = os.path.exists(args.csv_output)
    with open(args.csv_output, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)

    print("\n" + "=" * 65)
    print(f"RESULTS FOR {adapter_path.name}:")
    print(f"Accuracy  : {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"FPRR (False Alarm Rejection): {fprr:.2f}% | FNIR: {fnir:.2f}%")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn} (Ambiguous: {ambiguous_count})")
    print(f"Average Latency: {avg_lat:.2f} ms | P95: {p95_lat:.2f} ms | Peak VRAM: {peak_gpu_mem} MB")
    print(f"Saved to: {args.csv_output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
