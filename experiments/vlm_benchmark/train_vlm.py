#!/usr/bin/env python3
"""
Unified VLM Fine-Tuning Pipeline (Experiments 2 & 3)
Supports 5 VLM models across 2 visual input contexts (10 configurations):
- Models: Qwen2-VL-2B, Qwen2-VL-7B, Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B
- Context A: Single-Image (Full Image Only with coordinate tokens)
- Context B: Dual-Image (Full Image + 20% Zoomed-Out Candidate Crop)

Reference: writing-block.md (Sections 4 & 5)
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model

# System Prompts & Formatting Tokens
SYSTEM_PROMPT = "<|im_start|>system\nYou are a precise weapon detection verification engine.<|im_end|>\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>"

# Model Registry
VLM_REGISTRY = {
    "qwen2-2b": {
        "name": "Qwen2-VL-2B",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "Qwen2-VL",
        "model_class": "Qwen2VLForConditionalGeneration",
    },
    "qwen2-7b": {
        "name": "Qwen2-VL-7B",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "family": "Qwen2-VL",
        "model_class": "Qwen2VLForConditionalGeneration",
    },
    "qwen3-2b": {
        "name": "Qwen3-VL-2B",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "family": "Qwen3-VL",
        "model_class": "Qwen3VLForConditionalGeneration",
    },
    "qwen3-4b": {
        "name": "Qwen3-VL-4B",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "Qwen3-VL",
        "model_class": "Qwen3VLForConditionalGeneration",
    },
    "qwen3-8b": {
        "name": "Qwen3-VL-8B",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "Qwen3-VL",
        "model_class": "Qwen3VLForConditionalGeneration",
    },
}

# Default dataset and annotation discovery
def find_default_annotation_files():
    possible_roots = [
        Path("Dataset Annotation-20260807T120404Z-1-001-20261117T042214Z-1-001/Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation"),
        Path("Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation"),
        Path("Dataset Annotation"),
        Path("."),
    ]
    for root in possible_roots:
        train_p = root / "train.json"
        valid_p = root / "valid.json"
        test_p = root / "test.json"
        if train_p.is_file() and valid_p.is_file():
            return str(train_p), str(valid_p), str(test_p)
    return "train.json", "valid.json", "test.json"

DEFAULT_TRAIN_JSON, DEFAULT_VALID_JSON, DEFAULT_TEST_JSON = find_default_annotation_files()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision-Language Model Validator for Weapon Detection")
    parser.add_argument(
        "--model-key",
        type=str,
        default="qwen3-8b",
        choices=list(VLM_REGISTRY.keys()),
        help=f"Model identifier from registry: {list(VLM_REGISTRY.keys())}",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Custom HuggingFace model ID (overrides model-key)")
    parser.add_argument(
        "--context-mode",
        type=str,
        default="dual",
        choices=["single", "dual"],
        help="Visual input context: 'single' (Full Image Only) or 'dual' (Full Image + 20 percent Zoomed-Out Crop)",
    )
    parser.add_argument("--train-json", type=str, default=DEFAULT_TRAIN_JSON, help="Path to train annotations JSON")
    parser.add_argument("--valid-json", type=str, default=DEFAULT_VALID_JSON, help="Path to valid annotations JSON")
    parser.add_argument("--test-json", type=str, default=DEFAULT_TEST_JSON, help="Path to test annotations JSON")
    parser.add_argument("--data-dir", type=str, default="2-class-weapon-dataset-3", help="Base dataset directory for images")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fine-tuned LoRA adapter")
    parser.add_argument("--output-style", type=str, default="cot", choices=["cot", "binary"], help="'cot' (Reasoning + Final Decision) or 'binary' (WEAPON / NOT_WEAPON)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU device")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--margin-ratio", type=float, default=0.20, help="Zoom-out margin for cropped context in dual mode (default 0.20 = 20 percent)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging steps interval")
    parser.add_argument("--save-steps", type=int, default=200, help="Evaluation and checkpoint save interval")
    parser.add_argument("--standardize-labels", action="store_true", default=False, help="Force standardized single-word binary targets (WEAPON / NOT_WEAPON)")
    return parser.parse_args()

def resolve_image_path(image_path, fallback_dir="2-class-weapon-dataset-3"):
    """
    Robust image resolver that checks:
    1. Direct candidate path
    2. Stripped 'dataset/' prefix (common in roboflow VLM exports)
    3. Path relative to fallback_dir (e.g. 2-class-weapon-dataset-3)
    4. Basename directly in fallback_dir
    5. Dataset splits (train/valid/test) across 2-class and 1-class dataset directories
    """
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate

    # Strip leading 'dataset' if present
    stripped_candidate = Path(*candidate.parts[1:]) if (candidate.parts and candidate.parts[0] == "dataset") else candidate

    search_dirs = [
        Path(fallback_dir),
        Path("2-class-weapon-dataset-3"),
        Path("1-class-weapon-dataset-2"),
        Path("."),
    ]

    for s_dir in search_dirs:
        # direct relative with stripped
        cand1 = s_dir / stripped_candidate
        if cand1.is_file():
            return cand1

        cand2 = s_dir / candidate
        if cand2.is_file():
            return cand2

        # splits (train/valid/test)
        for split in ("train", "valid", "test"):
            split_cand = s_dir / split / "images" / candidate.name
            if split_cand.is_file():
                return split_cand

        # basename in dir
        b_name = s_dir / candidate.name
        if b_name.is_file():
            return b_name

    raise FileNotFoundError(f"Could not resolve image path: {image_path}")

def parse_box_coords(text_content):
    """
    Extracts normalized coordinates (ymin, xmin, ymax, xmax) in 0-1000 scale from <box>(ymin,xmin,ymax,xmax)</box>.
    """
    match = re.search(r"<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>", text_content)
    if match:
        return tuple(map(int, match.groups()))
    return None

def create_visual_inputs(image_path, box_coords, context_mode="dual", margin_ratio=0.20):
    """
    Creates visual input image(s) and prompt tokens based on context configuration:
    - Context A ('single'): Full Image Only.
    - Context B ('dual'): Full Image with 3px red bounding box + 20% zoomed-out crop.
    """
    img_full = Image.open(image_path).convert("RGB")
    width, height = img_full.size

    if context_mode == "single":
        # Configuration A: Full Image Only
        return [img_full], "<|vision_start|><|image_pad|><|vision_end|>"
    
    # Configuration B: Full Image + Candidate Bounding-Box Crop
    img1 = img_full.copy()
    draw = ImageDraw.Draw(img1)

    if box_coords:
        ymin_1000, xmin_1000, ymax_1000, xmax_1000 = box_coords
        xmin_px = (xmin_1000 / 1000.0) * width
        ymin_px = (ymin_1000 / 1000.0) * height
        xmax_px = (xmax_1000 / 1000.0) * width
        ymax_px = (ymax_1000 / 1000.0) * height

        # Draw 3-pixel red outline
        for offset in range(3):
            draw.rectangle(
                [xmin_px - offset, ymin_px - offset, xmax_px + offset, ymax_px + offset],
                outline=(255, 0, 0),
            )

        # Compute crop with expanded margin
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

    image_tokens = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
    return [img1, img2], image_tokens

def build_prompt_and_images(example, data_dir, context_mode="dual", add_generation_prompt=False, margin_ratio=0.20, standardize_labels=False, output_style="cot"):
    """
    Constructs the model input prompt and image inputs.
    """
    messages = example["messages"]
    user_message = next(msg for msg in messages if msg["role"] == "user")
    assistant_message = next((msg for msg in messages if msg["role"] == "assistant"), None)

    raw_image_path = None
    original_text = ""
    for item in user_message["content"]:
        if item.get("type") == "image":
            raw_image_path = item["image"]
        elif item.get("type") == "text":
            original_text += item["text"]

    resolved_path = resolve_image_path(raw_image_path, data_dir)
    box_coords = parse_box_coords(original_text)

    images, image_tokens = create_visual_inputs(
        resolved_path, box_coords, context_mode=context_mode, margin_ratio=margin_ratio
    )

    if context_mode == "dual":
        instruction_header = (
            "Picture 1 shows the full image with the candidate object highlighted in a bounding box. "
            "Picture 2 shows a cropped, 20% zoomed-out view of the object for detailed inspection. "
            "Analyze the object located inside the bounding box and determine whether the object is a real weapon or a visually similar non-weapon object."
        )
    else:
        coord_str = f"<box>{box_coords}</box>" if box_coords else "the highlighted coordinates"
        instruction_header = (
            f"Picture 1 shows the full scene. The detected candidate object is at {coord_str}. "
            "Analyze the object located inside the bounding box and determine whether the object is a real weapon or a visually similar non-weapon object."
        )

    user_content = f"{instruction_header}\n{original_text}"
    prompt = SYSTEM_PROMPT + f"<|im_start|>user\n{image_tokens}{user_content}<|im_end|>\n"

    if add_generation_prompt:
        return prompt + ASSISTANT_PREFIX, images

    # Build target assistant label
    assistant_raw = "".join(
        item.get("text", "")
        for item in (assistant_message.get("content", []) if assistant_message is not None else [])
        if item.get("type") == "text"
    )

    if standardize_labels or output_style == "binary":
        upper = assistant_raw.upper()
        if "NON-WEAPON" in upper or "NOT_WEAPON" in upper or "NOT WEAPON" in upper or "FALSE ALARM" in upper:
            assistant_target = "NOT_WEAPON"
        elif "WEAPON" in upper:
            assistant_target = "WEAPON"
        else:
            assistant_target = assistant_raw.strip()
    else:
        # Default 'cot': Preserves natural Reasoning + Final Decision from Dataset Annotation
        assistant_target = assistant_raw

    full_text = prompt + ASSISTANT_PREFIX + assistant_target + ASSISTANT_SUFFIX
    return full_text, images

def main():
    args = parse_args()

    # Determine Model Meta
    if args.model_id:
        model_id = args.model_id
        model_name = Path(model_id).name
        model_class_name = "Qwen3VLForConditionalGeneration" if "Qwen3" in model_id else "Qwen2VLForConditionalGeneration"
    else:
        reg = VLM_REGISTRY[args.model_key]
        model_id = reg["model_id"]
        model_name = reg["name"]
        model_class_name = reg["model_class"]

    if args.output_dir is None:
        clean_key = args.model_key.replace("-", "_")
        args.output_dir = f"adapters/vlm/{clean_key}_{args.context_mode}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("VLM Fine-Tuning Execution Plan")
    print("=" * 65)
    print(f"Model Name    : {model_name}")
    print(f"Model ID      : {model_id}")
    print(f"Model Class   : {model_class_name}")
    print(f"Context Mode  : {args.context_mode.upper()} ({'Full Image Only' if args.context_mode == 'single' else 'Full Image + 20% Zoomed-Out Crop'})")
    print(f"Train Dataset : {args.train_json}")
    print(f"Valid Dataset : {args.valid_json}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"Effective BS  : {args.batch_size * args.grad_accum}")
    print(f"Learning Rate : {args.learning_rate}")
    print(f"Output Dir    : {args.output_dir}")
    print("=" * 65)

    # 1. Load Dataset
    print("Loading dataset splits...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_json,
            "validation": args.valid_json,
            "test": args.test_json,
        },
    )
    print(f"Loaded: {len(dataset['train'])} train, {len(dataset['validation'])} valid, {len(dataset['test'])} test")

    # 2. Configure 4-bit Quantization (NF4) for 16GB VRAM safety
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Instantiate Model (with download retry loop)
    print(f"Loading {model_name} in 4-bit precision...")
    max_retries = 5
    model = None
    processor = None

    for attempt in range(1, max_retries + 1):
        try:
            if model_class_name == "Qwen3VLForConditionalGeneration":
                from transformers import Qwen3VLForConditionalGeneration
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
            else:
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )

            processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=256 * 256,
                max_pixels=768 * 768,
            )
            break
        except Exception as e:
            print(f"\n[WARNING] Attempt {attempt}/{max_retries} failed to load {model_id}: {e}")
            if attempt < max_retries:
                wait_time = 15 * attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    # 4. Configure PEFT LoRA
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Data Collator
    def collate_fn(batch):
        prompt_texts = []
        full_texts = []
        batch_images = []

        for example in batch:
            prompt_text, example_images = build_prompt_and_images(
                example,
                args.data_dir,
                context_mode=args.context_mode,
                add_generation_prompt=True,
                margin_ratio=args.margin_ratio,
                standardize_labels=args.standardize_labels,
                output_style=args.output_style,
            )
            full_text, _ = build_prompt_and_images(
                example,
                args.data_dir,
                context_mode=args.context_mode,
                add_generation_prompt=False,
                margin_ratio=args.margin_ratio,
                standardize_labels=args.standardize_labels,
                output_style=args.output_style,
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            batch_images.append(example_images)

        tokenized_full = processor(
            text=full_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        tokenized_prompt = processor(
            text=prompt_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        labels = tokenized_full["input_ids"].clone()
        prompt_mask_lengths = tokenized_prompt["attention_mask"].sum(dim=1)
        for i, plen in enumerate(prompt_mask_lengths.tolist()):
            if plen > 0:
                labels[i, : int(plen)] = -100

        tokenized_full["labels"] = labels
        return tokenized_full

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=0,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    t0 = time.time()
    print("\nStarting VLM fine-tuning...")
    train_result = trainer.train()
    training_duration_sec = round(time.time() - t0, 2)

    # 7. Save Adapter and Processor
    print(f"\nSaving fine-tuned adapter and processor to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save Experiment Run Metadata for Reproducibility
    metadata = {
        "model_name": model_name,
        "model_id": model_id,
        "model_class": model_class_name,
        "context_mode": args.context_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "margin_ratio": args.margin_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_time_sec": training_duration_sec,
        "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
    }
    with open(os.path.join(args.output_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training completed in {training_duration_sec:.1f}s. Adapter saved to {args.output_dir}!")

if __name__ == "__main__":
    main()
