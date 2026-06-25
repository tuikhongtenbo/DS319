"""
Main dispatcher entrypoint for training, inference, and evaluation.
Delegates execution to model-specific scripts to avoid cross-compatibility errors.
"""

import argparse
import sys
from pathlib import Path

from src.configs.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.utils.seed import set_seed
from src.utils.io import load_jsonl
from src.metrics.metrics import calculate_spatial_metrics

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SpatialMQA Unified Dispatcher Entrypoint")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer", "eval"], help="Execution mode")
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    parser.add_argument("--jsonl_dir", type=str, help="Path to jsonl dataset directory or file")
    parser.add_argument("--out_checkpoint", type=str, help="Path to save or load best checkpoint")
    parser.add_argument("--out_results", type=str, help="Path to save logs and predictions")
    parser.add_argument("--api_key", type=str, default="", help="API key for GPT/Qwen")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots for API models")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    return parser.parse_args()

def run_train(args, config: ExperimentConfig):
    model_type = config.model.model_type.lower()
    logger.info(f"Dispatching training for model type: {model_type}")
    
    if "blip-vqa-base" in model_type:
        from src.trainers.train_blip import run_train as blip_train
        blip_train(args, config)
    elif "blip2" in model_type:
        from src.trainers.train_blip2 import run_train as blip2_train
        blip2_train(args, config)
    elif "instructblip" in model_type:
        from src.trainers.train_instructblip import run_train as instructblip_train
        instructblip_train(args, config)
    elif "idefics" in model_type:
        from src.trainers.train_idefics import run_train as idefics_train
        idefics_train(args, config)
    elif "spacellava" in model_type:
        from src.trainers.train_spacellava import run_train as spacellava_train
        spacellava_train(args, config)
    elif "llava" in model_type:
        from src.trainers.train_llava import run_train as llava_train
        llava_train(args, config)
    elif "mplug" in model_type:
        from src.trainers.train_mplug_owl import run_train as mplug_train
        mplug_train(args, config)
    else:
        raise ValueError(f"Unsupported model type for training: {model_type}")

def run_infer(args, config: ExperimentConfig):
    model_type = config.model.model_type.lower()
    logger.info(f"Dispatching inference for model type: {model_type}")
    
    # 1. Closed-source API Models
    if "gpt-4" in model_type:
        logger.info(f"Initializing GPT Predictor ({args.shots}-shot)")
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        if data_path.is_dir():
            target_data_path = data_path / "test.jsonl"
            train_data_path = data_path / "train.jsonl"
        else:
            target_data_path = data_path
            train_data_path = data_path.parent / "train.jsonl"
            
        if args.shots >= 1:
            from src.inference.inference_gpt_1_shot import GPTOneShotPredictor
            predictor = GPTOneShotPredictor(
                model_name=model_type,
                api_key=args.api_key,
                train_data_path=str(train_data_path),
                image_dir=args.image_dir or config.dataset.image_dir
            )
        else:
            from src.inference.inference_gpt_0_shot import GPTZeroShotPredictor
            predictor = GPTZeroShotPredictor(model_name=model_type, api_key=args.api_key)
            
        # Run unified API loop
        run_api_inference_loop(args, config, predictor, target_data_path)
        
    elif "qwen" in model_type:
        logger.info(f"Initializing Qwen API Predictor ({args.shots}-shot)")
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        if data_path.is_dir():
            target_data_path = data_path / "test.jsonl"
            train_data_path = data_path / "train.jsonl"
        else:
            target_data_path = data_path
            train_data_path = data_path.parent / "train.jsonl"
            
        from src.inference.inference_qwen import QwenPredictor
        predictor = QwenPredictor(
            model_name=model_type,
            api_key=args.api_key,
            shots=args.shots,
            train_data_path=str(train_data_path) if args.shots >= 1 else None,
            image_dir=args.image_dir or config.dataset.image_dir if args.shots >= 1 else None
        )
        
        # Run unified API loop
        run_api_inference_loop(args, config, predictor, target_data_path)
        
    # 2. Open-source Models
    elif "blip-vqa-base" in model_type:
        from src.inference.inference_blip import run_infer as blip_infer
        blip_infer(args, config)
    elif "blip2" in model_type:
        from src.inference.inference_blip2 import run_infer as blip2_infer
        blip2_infer(args, config)
    elif "instructblip" in model_type:
        from src.inference.inference_instructblip import run_infer as instructblip_infer
        instructblip_infer(args, config)
    elif "idefics" in model_type:
        from src.inference.inference_idefics import run_infer as idefics_infer
        idefics_infer(args, config)
    elif "spacellava" in model_type:
        from src.inference.inference_spacellava import run_infer as spacellava_infer
        spacellava_infer(args, config)
    elif "llava" in model_type:
        from src.inference.inference_llava import run_infer as llava_infer
        llava_infer(args, config)
    elif "mplug" in model_type:
        from src.inference.inference_mplug_owl import run_infer as mplug_infer
        mplug_infer(args, config)
    else:
        raise ValueError(f"Unsupported model type for inference: {model_type}")

def run_api_inference_loop(args, config, predictor, target_data_path):
    from tqdm import tqdm
    from src.utils.io import save_jsonl
    
    logger.info(f"Loading API test dataset from {target_data_path}")
    test_data = load_jsonl(str(target_data_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    
    predictions = []
    logger.info("Starting API inference...")
    for item in tqdm(test_data):
        image_path = image_dir / item["image"]
        question = item["question"]
        options = item.get("options", [])
        
        output = predictor.predict(str(image_path), question, options)
        
        result_item = {
            "id": item["id"],
            "result": 1 if output.lower() in item["answer"] or item["answer"] in output.lower() else 0,
            "output": output.lower(),
            "answer": item["answer"]
        }
        predictions.append(result_item)
        
    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    logger.info(f"Saving predictions to {out_path}")
    save_jsonl(predictions, str(out_path))
    
    run_eval(predictions)

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
