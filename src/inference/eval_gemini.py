"""
Evaluation script for Gemini 3.1 Flash-Lite on SpatialMQA.
Supports 0-shot and 1-shot inference with all metrics.
"""

import argparse
import logging
import time
from pathlib import Path

from ..datasets.preprocessing import build_result_record
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_json, save_jsonl
from .inference_gemini_0_shot import GeminiZeroShotPredictor
from .inference_gemini_1_shot import GeminiOneShotPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gemini 3.1 Flash-Lite on SpatialMQA")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-flash-lite", help="Gemini model name")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="outputs/gemini", help="Output directory")
    parser.add_argument("--shot", type=int, default=0, choices=[0, 1], help="0-shot or 1-shot")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    if args.shot == 0:
        predictor = GeminiZeroShotPredictor(args.model_name, args.api_key)
        output_file = output_dir / f"gemini_{args.model_name}_0shot_results.jsonl"
    else:
        predictor = GeminiOneShotPredictor(args.model_name, args.api_key, args.image_dir)
        output_file = output_dir / f"gemini_{args.model_name}_1shot_results.jsonl"

    # Load test data
    test_data = load_jsonl(args.data_path)
    logger.info(f"Loaded {len(test_data)} test samples from {args.data_path}")

    # Run inference
    results = []
    for i, item in enumerate(test_data):
        image_path = str(Path(args.image_dir) / item["image"])
        question = item["question"]
        options = item.get("options", [])

        logger.info(f"[{i+1}/{len(test_data)}] Processing sample {item.get('id', i)}")

        output = predictor.predict(image_path, question, options)
        record = build_result_record(item, i, output)
        results.append(record)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(test_data)} samples")

        time.sleep(args.delay)  # Rate limiting

    # Save results
    save_jsonl(results, output_file)
    logger.info(f"Saved results to {output_file}")

    # Calculate metrics
    metrics = calculate_spatial_metrics(results)

    # Print metrics
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Shot: {args.shot}")
    logger.info(f"Total samples: {len(results)}")
    logger.info("-" * 50)
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info("-" * 50)
    logger.info(f"Accuracy-X (left/right): {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy-Y (on/below):  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy-Z (front/behind): {metrics['accuracy_z']:.4f}")
    logger.info("=" * 50)

    # Save metrics
    metrics_file = output_dir / f"gemini_{args.model_name}_{args.shot}shot_metrics.json"
    metrics_output = {
        "model": args.model_name,
        "shot": args.shot,
        "total_samples": len(results),
        "metrics": metrics,
    }
    save_json(metrics_output, metrics_file)
    logger.info(f"Saved metrics to {metrics_file}")

    return metrics


if __name__ == "__main__":
    main()
