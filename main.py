"""
Main entrypoint for training, inference, and evaluation.
"""

import argparse
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.configs.config import ExperimentConfig
from src.datasets.dataset import SpatialMQADataset
from src.models.builder import build_model_and_processor
from src.trainers.trainer import Trainer
from src.trainers.llava_trainer import LLaVATrainerWrapper
from src.inference.predictor import OpenSourcePredictor, APIPredictor
from src.metrics.metrics import calculate_spatial_metrics
from src.utils.io import load_jsonl, save_jsonl
from src.utils.seed import set_seed
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SpatialMQA Unified Entrypoint")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer", "eval"], help="Execution mode")
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    parser.add_argument("--jsonl_dir", type=str, help="Path to jsonl dataset directory or file")
    parser.add_argument("--out_checkpoint", type=str, help="Path to save best checkpoint")
    parser.add_argument("--out_results", type=str, help="Path to save logs and predictions")
    parser.add_argument("--api_key", type=str, default="", help="API key for GPT/Qwen")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots for API models")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    return parser.parse_args()

def run_train(args, config):
    set_seed(config.training.seed)
    
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint
    
    if "llava" in config.model.model_type.lower() or "spacellava" in config.model.model_type.lower():
        logger.info("Using LLaVA training wrapper...")
        trainer = LLaVATrainerWrapper(
            data_path=args.jsonl_dir or config.dataset.data_path,
            image_dir=args.image_dir or config.dataset.image_dir,
            output_dir=str(out_checkpoint),
            model_path=config.model.model_name_or_path,
            is_spacellava="spacellava" in config.model.model_type.lower()
        )
        trainer.prepare_and_generate_script()
    else:
        logger.info("Building model and processor...")
        model, processor = build_model_and_processor(config.model)
        
        batch_size = args.batch_size or config.training.batch_size
        
        # Assume jsonl_dir is a file path if it ends with jsonl, otherwise we append train.jsonl
        # In a real scenario, you'd split train/val
        data_path = Path(args.jsonl_dir or config.dataset.data_path)
        if data_path.is_dir():
            train_path = data_path / "train.jsonl"
            val_path = data_path / "dev.jsonl" 
            if not val_path.exists():
                val_path = train_path # Fallback
        else:
            train_path = data_path
            val_path = data_path
            
        logger.info(f"Loading train dataset from {train_path}")
        train_dataset = SpatialMQADataset(
            data_path=str(train_path),
            image_dir=args.image_dir or config.dataset.image_dir,
            processor=processor,
            max_samples=config.dataset.max_samples,
            is_training=True
        )
        
        logger.info(f"Loading valid dataset from {val_path}")
        valid_dataset = SpatialMQADataset(
            data_path=str(val_path),
            image_dir=args.image_dir or config.dataset.image_dir,
            processor=processor,
            max_samples=config.dataset.max_samples,
            is_training=True
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            processor=processor,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            config=config.training,
            device=device,
            out_checkpoint=out_checkpoint,
            out_results=out_results
        )
        trainer.train()

def run_infer(args, config):
    model_type = config.model.model_type.lower()
    
    if model_type in ["gpt-4o", "qwen-3.6"]:
        logger.info(f"Initializing API Predictor for {model_type}")
        predictor = APIPredictor(model_name=model_type, api_key=args.api_key, shots=args.shots)
    else:
        logger.info(f"Building Open Source Model: {model_type}")
        model, processor = build_model_and_processor(config.model)
        # If inferencing a trained model, load checkpoint
        if args.out_checkpoint and Path(args.out_checkpoint).exists():
            logger.info(f"Loading trained weights from {args.out_checkpoint}")
            model.load_adapter(str(Path(args.out_checkpoint) / "best_model")) # Load LoRA
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = OpenSourcePredictor(model=model, processor=processor, device=device)
        
    data_path = args.jsonl_dir or config.dataset.data_path
    logger.info(f"Loading test dataset from {data_path}")
    test_data = load_jsonl(data_path)
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    
    predictions = []
    
    logger.info("Starting inference...")
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
    
    # Optionally run eval immediately
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
