#!/usr/bin/env python3
import os
import sys
import gc
import csv
import argparse
import torch
import cv2
cv2.setNumThreads(0)
from roboflow import Roboflow
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train all YOLO variants with automatic batch size lowering on OOM.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training and testing")
    parser.add_argument(
        "--models", 
        type=str, 
        default="yolo12m.pt,yolo12l.pt,yolo12x.pt", 
        help="Comma-separated list of YOLO weights to train"
    )
    parser.add_argument("--csv-output", type=str, default="test_results.csv", help="Path to output test results CSV")
    parser.add_argument("--dataset-dir", type=str, default="1-class-weapon-dataset-2", help="Directory where dataset is located")
    parser.add_argument("--api-key", type=str, default="Q1flzyg1J8Z2OMweLXf3", help="Roboflow API key")
    parser.add_argument("--workspace", type=str, default="weapon-mpr3p", help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default="1-class-weapon-dataset", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=2, help="Roboflow version number")
    parser.add_argument("--batch", type=int, default=None, help="Global override for initial batch size")
    parser.add_argument("--device", type=str, default="", help="Device to run on (e.g. 0, cpu)")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for data loading (0 avoids Windows multiprocessing issues)")
    return parser.parse_args()

def check_and_download_dataset(args):
    dataset_dir = args.dataset_dir
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    
    if os.path.exists(dataset_dir) and os.path.exists(data_yaml):
        print(f"Dataset already exists at: {os.path.abspath(dataset_dir)}")
        return os.path.abspath(data_yaml)
        
    print(f"Dataset not found at {dataset_dir}. Downloading from Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov12")
    downloaded_yaml = os.path.join(dataset.location, "data.yaml")
    if os.path.exists(downloaded_yaml):
        return os.path.abspath(downloaded_yaml)
    else:
        raise FileNotFoundError(f"Could not find data.yaml in downloaded dataset location: {dataset.location}")

def get_default_batch(model_name):
    # Match the default initial batch sizes from the notebook or standard defaults
    if "12n" in model_name:
        return 16
    elif "12s" in model_name:
        return 32
    elif "12m" in model_name:
        return 16
    elif "12l" in model_name:
        return 8
    elif "12x" in model_name:
        return 4
    return 32

def train_one_model(model_name, data_yaml_path, initial_batch, epochs, imgsz, device, workers):
    batch = initial_batch
    run_name = model_name.replace(".pt", "")
    
    while batch >= 1:
        print("\n" + "="*80)
        print(f"Starting training attempt: {model_name} with batch={batch}")
        print("="*80 + "\n")
        
        # Explicitly clean up memory before model loading/initialization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Load a fresh model instance inside the OOM protection block
            model = YOLO(model_name)
            
            # Train the model
            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name=run_name,
                device=device,
                exist_ok=True,
                workers=workers
            )
            print(f"Training successfully completed for {model_name} with batch={batch}.")
            return model, batch
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda oom" in error_msg or "oom" in error_msg:
                print(f"\n[WARNING] CUDA Out of Memory detected during training of {model_name} with batch={batch}.")
                new_batch = batch // 2
                if new_batch < 1:
                    print(f"[ERROR] Batch size cannot be reduced below 1. Training failed for {model_name}.")
                    raise RuntimeError(f"OOM at batch=1 for {model_name}") from e
                
                print(f"Reducing batch size from {batch} to {new_batch} and retrying...\n")
                batch = new_batch
                
                # Cleanup model reference
                if 'model' in locals():
                    del model
            else:
                # Re-raise any other errors (dataset not found, syntax error, etc.)
                raise e

def main():
    args = parse_args()
    
    # Enforce GPU-only training
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA is not available. Only GPU training is allowed.")
    if args.device.lower() == "cpu":
        raise ValueError("Device cannot be set to 'cpu'. Only GPU training is allowed.")
    
    # 1. Check and download/locate dataset
    data_yaml_path = check_and_download_dataset(args)
    
    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Models to train: {models_list}")
    
    # Load existing results to append to
    all_results = load_existing_results(args.csv_output)
    
    for model_name in models_list:
        # Determine initial batch size
        initial_batch = args.batch if args.batch is not None else get_default_batch(model_name)
        
        status = "Success"
        error_msg = ""
        succeeded_batch = None
        metrics_dict = {}
        
        try:
            # 2. Train with OOM protection
            model, succeeded_batch = train_one_model(
                model_name=model_name,
                data_yaml_path=data_yaml_path,
                initial_batch=initial_batch,
                epochs=args.epochs,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers
            )
            
            # 3. Evaluate on test split using best weights
            print(f"\nEvaluating {model_name} on 'test' split...")
            best_weights_path = ""
            if hasattr(model, 'trainer') and model.trainer is not None:
                best_weights_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
                
            # Clean up training model reference and clear cache before eval load
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if best_weights_path and os.path.exists(best_weights_path):
                print(f"Loading best weights from {best_weights_path} for testing.")
                eval_model = YOLO(best_weights_path)
            else:
                print(f"Best weights path not resolved or file not found. Reloading base model weights.")
                eval_model = YOLO(model_name)
                
            val_batch = 16  # Default initial validation batch size
            val_success = False
            while val_batch >= 1:
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    print(f"Running evaluation with val_batch={val_batch}...")
                    metrics = eval_model.val(
                        data=data_yaml_path,
                        split="test",
                        imgsz=args.imgsz,
                        batch=val_batch,
                        device=args.device,
                        workers=args.workers
                    )
                    metrics_dict = metrics.results_dict
                    # Add map75 metric explicitly
                    if hasattr(metrics, 'box') and hasattr(metrics.box, 'map75'):
                        metrics_dict['metrics/mAP75(B)'] = float(metrics.box.map75)
                    val_success = True
                    print(f"Evaluation results for {model_name}: {metrics_dict}")
                    break
                except Exception as eval_e:
                    eval_err = str(eval_e).lower()
                    if "out of memory" in eval_err or "cuda oom" in eval_err or "oom" in eval_err:
                        new_val_batch = val_batch // 2
                        print(f"\n[WARNING] CUDA Out of Memory during evaluation of {model_name} with val_batch={val_batch}.")
                        if new_val_batch < 1:
                            print(f"[ERROR] Validation batch size cannot be reduced below 1. Validation failed.")
                            status = "Success (Val Failed)"
                            error_msg = f"Validation failed due to OOM: {eval_e}"
                            break
                        print(f"Reducing validation batch size from {val_batch} to {new_val_batch} and retrying...\n")
                        val_batch = new_val_batch
                    else:
                        print(f"[ERROR] Validation failed with non-OOM error: {eval_e}")
                        status = "Success (Val Failed)"
                        error_msg = f"Validation failed: {eval_e}"
                        break
            
        except Exception as e:
            status = "Failed"
            error_msg = str(e)
            print(f"[ERROR] Failed training/evaluating {model_name}: {e}")
            
        # Compile row for CSV
        row = {
            "Model": model_name,
            "Status": status,
            "Succeeded_Batch_Size": succeeded_batch if succeeded_batch else "N/A",
            "Error_Message": error_msg
        }
        # Add all evaluation metrics to the row
        for k, v in metrics_dict.items():
            # Clean up key names for better CSV header readability
            clean_key = k.replace("metrics/", "").replace("(B)", "")
            row[clean_key] = v
            
        # Clean up evaluation model references to free GPU memory
        if 'eval_model' in locals():
            del eval_model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_e:
            print(f"[WARNING] CUDA cleanup failed: {cleanup_e}")
            
        update_or_append_result(all_results, row)
        
        # Save CSV incrementally in case the script is interrupted later
        save_csv(all_results, args.csv_output)
        
    print(f"\nAll models finished. Final results written to {args.csv_output}")

def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return []
    results = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields back to float/int if possible
                cleaned_row = {}
                for k, v in row.items():
                    if v == "":
                        cleaned_row[k] = ""
                    else:
                        try:
                            if "." in v:
                                cleaned_row[k] = float(v)
                            else:
                                cleaned_row[k] = int(v)
                        except ValueError:
                            cleaned_row[k] = v
                results.append(cleaned_row)
        print(f"Loaded {len(results)} existing results from {csv_path}")
    except Exception as e:
        print(f"Warning: Could not read existing CSV {csv_path}: {e}")
    return results

def update_or_append_result(results, new_row):
    for i, r in enumerate(results):
        if r.get("Model") == new_row.get("Model"):
            results[i] = new_row
            return
    results.append(new_row)

def save_csv(results, csv_path):
    if not results:
        return
        
    # Get union of all keys across all dicts
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
        
    # Define desired header ordering
    primary_headers = ["Model", "Status", "Succeeded_Batch_Size", "Error_Message"]
    other_headers = sorted(list(all_keys - set(primary_headers)))
    fieldnames = primary_headers + other_headers
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                # Fill missing keys with empty string
                row_to_write = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row_to_write)
        print(f"Updated CSV results at {csv_path}")
    except Exception as e:
        print(f"Error writing CSV to {csv_path}: {e}")

if __name__ == "__main__":
    main()
