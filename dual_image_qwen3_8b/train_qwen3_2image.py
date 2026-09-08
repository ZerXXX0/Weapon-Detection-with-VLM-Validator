#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-8B VLM with Dual Images (Full Image + 20% Zoomed-Out Crop)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base Qwen3-VL model ID")
    parser.add_argument("--data-dir", type=str, default="2-class-weapon-dataset-3", help="Base dataset directory")
    parser.add_argument("--train-json", type=str, default="Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation/train.json", help="Path to train.json")
    parser.add_argument("--valid-json", type=str, default="Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation/valid.json", help="Path to valid.json")
    parser.add_argument("--test-json", type=str, default="Dataset Annotation-20260807T120404Z-1-001/Dataset Annotation/test.json", help="Path to test.json")
    parser.add_argument("--output-dir", type=str, default="./qwen3_vl_model_2image", help="Directory to save trained model adapter")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--margin-ratio", type=float, default=0.20, help="Zoom-out margin ratio for crop (default 0.20 = 20%% zoom out)")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save and eval steps")
    return parser.parse_args()

def resolve_image_path(image_path, data_dir):
    data_dir_path = Path(data_dir)
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate

    dataset_candidate = data_dir_path / candidate
    if dataset_candidate.is_file():
        return dataset_candidate

    basename_candidate = data_dir_path / candidate.name
    if basename_candidate.is_file():
        return basename_candidate

    for split_name in ("train", "valid", "test"):
        split_candidate = data_dir_path / split_name / "images" / candidate.name
        if split_candidate.is_file():
            return split_candidate

    raise FileNotFoundError(f"Could not resolve image path: {image_path}")

def parse_box_coords(text_content):
    """
    Extracts normalized coordinates (ymin, xmin, ymax, xmax) in 0-1000 scale from <box>(ymin,xmin,ymax,xmax)</box>.
    """
    match = re.search(r"<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>", text_content)
    if match:
        ymin, xmin, ymax, xmax = map(int, match.groups())
        return ymin, xmin, ymax, xmax
    return None

def create_dual_images(image_path, box_coords, margin_ratio=0.20):
    """
    Generates:
    - Image 1: Full image with bounding box annotation drawn in red.
    - Image 2: Cropped version of the bounding box with 20% expanded margin (zoom-out).
    """
    img_full = Image.open(image_path).convert("RGB")
    width, height = img_full.size

    # Image 1: Full image with red bounding box
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
                outline=(255, 0, 0)
            )

        # Image 2: Cropped ROI with 20% expanded margin
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

    return img1, img2

def build_prompt_and_images(example, data_dir, add_generation_prompt, margin_ratio=0.20):
    messages = example["messages"]
    user_message = next(message for message in messages if message["role"] == "user")
    assistant_message = next((message for message in messages if message["role"] == "assistant"), None)

    raw_image_path = None
    original_text = ""
    for content_item in user_message["content"]:
        if content_item["type"] == "image":
            raw_image_path = content_item["image"]
        elif content_item["type"] == "text":
            original_text += content_item["text"]

    resolved_path = resolve_image_path(raw_image_path, data_dir)
    box_coords = parse_box_coords(original_text)

    img1, img2 = create_dual_images(resolved_path, box_coords, margin_ratio=margin_ratio)
    images = [img1, img2]

    # Two image vision placeholders
    image_tokens = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
    
    # Dual-image instructions context
    dual_instruction = (
        "Picture 1 shows the full image with the candidate object highlighted in a bounding box. "
        "Picture 2 shows a cropped, 20% zoomed-out view of the object for detailed inspection. "
    )
    user_text = dual_instruction + original_text

    prompt = SYSTEM_PROMPT + f"<|im_start|>user\n{image_tokens}{user_text}<|im_end|>\n"

    if add_generation_prompt:
        return prompt + ASSISTANT_PREFIX, images

    assistant_text = "".join(
        content_item["text"]
        for content_item in (assistant_message["content"] if assistant_message is not None else [])
        if content_item.get("type") == "text"
    )
    return prompt + ASSISTANT_PREFIX + assistant_text + ASSISTANT_SUFFIX, images

def main():
    args = parse_args()

    print(f"Loading datasets from JSON files...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_json,
            "validation": args.valid_json,
            "test": args.test_json,
        },
    )

    print(f"Loading Qwen3-VL base model '{args.model_id}' in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(args.model_id)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    def collate_fn(batch):
        prompt_texts = []
        full_texts = []
        batch_images = []

        for example in batch:
            prompt_text, example_images = build_prompt_and_images(
                example, args.data_dir, add_generation_prompt=True, margin_ratio=args.margin_ratio
            )
            full_text, _ = build_prompt_and_images(
                example, args.data_dir, add_generation_prompt=False, margin_ratio=args.margin_ratio
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            batch_images.append(example_images)

        tokenized_full = processor(
            text=full_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )

        tokenized_prompt = processor(
            text=prompt_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )

        labels = tokenized_full["input_ids"].clone()
        prompt_mask_lengths = tokenized_prompt["attention_mask"].sum(dim=1)
        for i, plen in enumerate(prompt_mask_lengths.tolist()):
            if plen > 0:
                labels[i, : int(plen)] = -100

        tokenized_full["labels"] = labels
        return tokenized_full

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=0,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=200,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
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

    print("Starting Qwen3-8B 2-Image Training...")
    trainer.train()

    print(f"Saving fine-tuned model and processor to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
