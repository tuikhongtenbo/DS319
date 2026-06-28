"""
Main dispatcher entrypoint for training, inference, and evaluation.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.configs.config import ExperimentConfig
from src.datasets.preprocessing import build_result_record, resolve_test_path
from src.metrics.metrics import calculate_spatial_metrics
from src.utils.hf_hub import configure_hf_hub_downloads
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

configure_hf_hub_downloads()

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SpatialMQA Unified Dispatcher")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer", "eval"],
    )
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    parser.add_argument("--jsonl_dir", type=str, help="Path to jsonl dataset directory or file")
    parser.add_argument(
        "--out_checkpoint",
        type=str,
        help="Path to save or load best checkpoint",
    )
    parser.add_argument("--out_results", type=str, help="Path to save logs and predictions")
    parser.add_argument("--api_key", type=str, default="", help="API key for GPT/Qwen API models")
    parser.add_argument("--model_name", type=str, default="", help="Override API model name")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots for API models")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for API inference (GPT/Qwen).",
    )
    parser.add_argument("--vllm_host", type=str, default=None, help="vLLM server URL (e.g., http://localhost:8000)")
    return parser.parse_args()


# Training dispatchers

def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)
    model_type = config.model.model_type.lower()
    logger.info(f"Dispatching training for: {model_type}")

    if "blip-vqa-base" in model_type:
        from src.trainers.train_blip import run_train as blip_train
        blip_train(args, config)
    elif "blip2" in model_type:
        from src.trainers.train_blip2 import run_train as blip2_train
        blip2_train(args, config)
    elif "spacellava" in model_type:
        from src.trainers.train_spacellava import run_train as spacellava_train
        spacellava_train(args, config)
    elif "llava" in model_type:
        from src.trainers.train_llava import run_train as llava_train
        llava_train(args, config)
    else:
        raise ValueError(f"Unsupported model type for training: {model_type}")


# Inference dispatchers

def run_infer(args, config: ExperimentConfig):
    model_type = config.model.model_type.lower()
    use_vllm = config.model.use_vllm
    logger.info(f"Dispatching inference for: {model_type} (vLLM={use_vllm})")

    # vLLM-accelerated paths
    if use_vllm:
        if "llava" in model_type or "spacellava" in model_type:
            from src.inference.inference_vllm_llava import run_infer as llava_vllm_infer
            llava_vllm_infer(args, config)
            return
        elif "qwen_vl" in model_type:
            from src.inference.inference_vllm_qwen import run_infer as qwen_vllm_infer
            qwen_vllm_infer(args, config)
            return
        else:
            logger.warning(
                f"vLLM mode requested for '{model_type}' but no script exists. "
                "Falling back to HuggingFace."
            )

    # HuggingFace / API paths
    if "gpt-4" in model_type:
        logger.info(f"Initializing GPT Predictor ({args.shots}-shot)")
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        target_data_path = resolve_test_path(data_path)
        train_data_path = (
            data_path / "train.jsonl"
            if data_path.is_dir()
            else data_path.parent / "train.jsonl"
        )

        if args.shots >= 1:
            from src.inference.inference_gpt_1_shot import GPTOneShotPredictor
            predictor = GPTOneShotPredictor(
                model_name=model_type,
                api_key=args.api_key,
                train_data_path=str(train_data_path),
                image_dir=args.image_dir or config.dataset.image_dir,
            )
        else:
            from src.inference.inference_gpt_0_shot import GPTZeroShotPredictor
            predictor = GPTZeroShotPredictor(model_name=model_type, api_key=args.api_key)

        run_api_inference_loop(args, config, predictor, target_data_path)
    elif "blip-vqa-base" in model_type:
        from src.inference.inference_blip import run_infer as blip_infer
        blip_infer(args, config)
    elif "blip2" in model_type:
        from src.inference.inference_blip2 import run_infer as blip2_infer
        blip2_infer(args, config)
    elif "instructblip" in model_type:
        from src.inference.inference_instructblip import run_infer as instructblip_infer
        instructblip_infer(args, config)
    elif "spacellava" in model_type:
        from src.inference.inference_spacellava import run_infer as spacellava_infer
        spacellava_infer(args, config)
    elif "gemini" in model_type:
        logger.info(f"Initializing Gemini Predictor ({args.shots}-shot)")
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        target_data_path = resolve_test_path(data_path)

        if args.shots >= 1:
            from src.inference.inference_gemini_1_shot import GeminiOneShotPredictor
            predictor = GeminiOneShotPredictor(
                model_name=model_type,
                api_key=args.api_key,
                image_dir=args.image_dir or config.dataset.image_dir,
            )
        else:
            from src.inference.inference_gemini_0_shot import GeminiZeroShotPredictor
            predictor = GeminiZeroShotPredictor(model_name=model_type, api_key=args.api_key)

        run_api_inference_loop(args, config, predictor, target_data_path)
    elif "llava" in model_type:
        from src.inference.inference_llava import run_infer as llava_infer
        llava_infer(args, config)
    elif "qwen_vl" in model_type:
        logger.info(f"Initializing Qwen Predictor ({args.shots}-shot)")
        qwen_model_name = args.model_name or config.model.model_name_or_path
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        target_data_path = resolve_test_path(data_path)
        train_data_path = (
            data_path / "train.jsonl"
            if data_path.is_dir()
            else data_path.parent / "train.jsonl"
        )

        if args.shots >= 1:
            from src.inference.inference_qwen_1_shot import QwenOneShotPredictor
            predictor = QwenOneShotPredictor(
                model_name=qwen_model_name,
                api_key=args.api_key,
                train_data_path=str(train_data_path),
                image_dir=args.image_dir or config.dataset.image_dir,
            )
        else:
            from src.inference.inference_qwen_0_shot import QwenZeroShotPredictor
            predictor = QwenZeroShotPredictor(
                model_name=qwen_model_name,
                api_key=args.api_key,
            )

        run_api_inference_loop(args, config, predictor, target_data_path)
    else:
        raise ValueError(f"Unsupported model type for inference: {model_type}")


def run_api_inference_loop(args, config, predictor, target_data_path):
    logger.info(f"Loading API test dataset from {target_data_path}")
    test_data = load_jsonl(str(target_data_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    num_workers = max(1, args.num_workers or 1)

    def predict_one(index, item):
        image_path = image_dir / item["image"]
        max_retries = 5
        output = "ERROR"

        for attempt in range(max_retries):
            try:
                output = predictor.predict(str(image_path), item["question"], item.get("options", []))
                break
            except Exception as e:
                is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Prediction failed after {max_retries} attempts: {e}")
                    break

        record = build_result_record(item, index, output)
        return index, item, output, record

    predictions = [None] * len(test_data)
    logger.info(f"Starting API inference with {num_workers} worker(s)...")

    if num_workers == 1:
        for index, item in enumerate(tqdm(test_data)):
            _, item, output, record = predict_one(index, item)
            predictions[index] = record
            logger.info(f"[{index+1}/{len(test_data)}] Q: {item['question'][:60]}... | A: {output}")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(predict_one, index, item)
                for index, item in enumerate(test_data)
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                index, item, output, record = future.result()
                predictions[index] = record
                logger.info(f"[{index+1}/{len(test_data)}] Q: {item['question'][:60]}... | A: {output}")

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    logger.info(f"Saving predictions to {out_path}")
    save_jsonl(predictions, str(out_path))

    run_eval(predictions, args)

# Evaluation

def run_eval(predictions=None, args=None):
    if predictions is None:
        out_results = Path(args.out_results) if args.out_results else Path("results")
        pred_path = out_results / "predictions.jsonl"
        logger.info(f"Loading predictions from {pred_path}")
        predictions = load_jsonl(str(pred_path))

    metrics = calculate_spatial_metrics(predictions)

    logger.info("--- Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X (Left/Right):    {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y (Above/Below):   {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z (Front/Behind):  {metrics['accuracy_z']:.4f}")

    if args is not None and args.out_results:
        metrics_path = Path(args.out_results) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")

    return metrics


# Entrypoint

def main():
    args = parse_args()

    config = None
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)

    if args.mode == "train":
        assert config is not None, "Training requires a config file."
        run_train(args, config)
    elif args.mode == "infer":
        assert config is not None, "Inference requires a config file."
        run_infer(args, config)
    elif args.mode == "eval":
        run_eval(args=args)


if __name__ == "__main__":
    main()
