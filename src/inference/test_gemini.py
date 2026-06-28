"""
Quick test script for Gemini 3.1 Flash-Lite with batch processing.
Runs in batches of 15 (free tier limit) with 60s pause between batches.
"""

import argparse
import time
from pathlib import Path

from src.datasets.preprocessing import build_result_record
from src.metrics.metrics import calculate_spatial_metrics
from src.utils.io import load_jsonl, save_json
from src.inference.inference_gemini_0_shot import GeminiZeroShotPredictor
from src.inference.inference_gemini_1_shot import GeminiOneShotPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite")
    parser.add_argument("--data_path", type=str, default="src/datasets/data/test_500.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/images/COCO2017")
    parser.add_argument("--shot", type=int, default=0, choices=[0, 1])
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--batch_pause", type=int, default=60)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None, help="Path to existing results JSON to resume")
    args = parser.parse_args()

    # Initialize predictor
    if args.shot == 0:
        predictor = GeminiZeroShotPredictor(args.model, args.api_key)
    else:
        predictor = GeminiOneShotPredictor(args.model, args.api_key, args.image_dir)

    # Load test data
    all_data = load_jsonl(args.data_path)
    test_data = all_data[args.start_idx:args.end_idx]
    print(f"Processing {len(test_data)} samples ({args.start_idx} to {args.end_idx})...")
    print(f"Batch size: {args.batch_size}, Pause between batches: {args.batch_pause}s")

    # Load existing results if resuming
    results = []
    if args.resume and Path(args.resume).exists():
        results = load_jsonl(args.resume)
        print(f"Resuming from {len(results)} existing results")

    # Calculate starting index
    start_from = len(results)
    if start_from >= len(test_data):
        print(f"Already completed all {len(test_data)} samples!")
        return

    test_data = test_data[start_from:]
    print(f"Starting from index {start_from}, {len(test_data)} samples remaining")

    # Run inference
    batch_count = 0
    for i, item in enumerate(test_data):
        # Check if we need a batch pause (every batch_size samples)
        if i > 0 and i % args.batch_size == 0:
            batch_count += 1
            print(f"\n{'='*50}")
            print(f"Batch {batch_count} complete ({i} samples done). Pausing {args.batch_pause}s...")
            print(f"Progress: {i}/{len(test_data)} ({100*i/len(test_data):.1f}%)")
            print(f"Estimated remaining: {(len(test_data)-i) * (args.batch_pause + 4) / 60:.1f} minutes")
            print(f"{'='*50}\n")
            time.sleep(args.batch_pause)

        image_path = str(Path(args.image_dir) / item["image"])
        output = predictor.predict(image_path, item["question"], item.get("options", []))
        record = build_result_record(item, start_from + i, output)
        results.append(record)
        print(f"[{start_from + i + 1}/{args.end_idx}] Q: {item['question'][:50]}... | A: {output}")

    # Calculate metrics
    metrics = calculate_spatial_metrics(results)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    output_path = f"outputs/gemini_{args.shot}shot_results.json"
    save_json(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
