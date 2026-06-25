"""
Inference script specialized for BLIP-1 (Salesforce/blip-vqa-base).
"""

import logging
from pathlib import Path
import json
import torch
from tqdm import tqdm
from PIL import Image

from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class BlipPredictor:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: list) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"
            
        options_str = "; ".join(options)
        prompt = f"Question: {question} Options: {options_str}"
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded

def run_infer(args, config: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Building BLIP-1 model and processor for inference...")
    processor = BlipProcessor.from_pretrained(config.model.model_name_or_path)
    
    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif config.model.load_in_4bit:
        kwargs["load_in_4bit"] = True
        
    model = BlipForQuestionAnswering.from_pretrained(config.model.model_name_or_path, **kwargs)
    
    # Check for LoRA checkpoints
    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if lora_path.exists():
            logger.info(f"Loading trained LoRA weights from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            logger.warning(f"Could not find best_model at {lora_path}. Using base model.")
            
    if not config.model.load_in_8bit and not config.model.load_in_4bit:
        model = model.to(device)
        
    predictor = BlipPredictor(model, processor, device)
    
    # Load dataset
    data_path = Path(args.jsonl_dir or config.dataset.data_path)
    if data_path.is_dir():
        target_path = data_path / "test.jsonl"
        if not target_path.exists():
            target_path = data_path / "dev.jsonl"
    else:
        target_path = data_path
        
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    
    predictions = []
    
    logger.info("Starting BLIP-1 inference...")
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
    
    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- BLIP-1 Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
