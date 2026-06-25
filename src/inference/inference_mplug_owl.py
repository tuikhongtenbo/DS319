"""
Inference script specialized for mPLUG-Owl.
"""

import os
import logging
from pathlib import Path
import json
import torch
from tqdm import tqdm
from PIL import Image

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MplugOwlPredictor:
    def __init__(self, model, processor, tokenizer, device):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
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
        
        prompt_text = f"You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {options_str}. \n Output:"
        
        # Conversation structure for mPLUG-Owl
        prompts = [
            f"The following is a conversation between a curious human and AI assistant.\nHuman: <image>\nHuman: {prompt_text}\nAI: "
        ]
        
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            res = self.model.generate(
                **inputs,
                do_sample=False,
                max_length=512,
                temperature=0.1
            )
            
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        return sentence.strip().rstrip('.')

def run_infer(args, config: ExperimentConfig):
    # Dynamic imports for mPLUG-Owl
    try:
        from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
        from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
        from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
        from peft import PeftModel
    except ImportError as e:
        logger.error(f"Failed to import mPLUG-Owl libraries: {e}")
        logger.warning("Skipping mPLUG-Owl inference run.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Building mPLUG-Owl model and processor for inference...")
    pretrained_ckpt = config.model.model_name_or_path
    
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    )
    
    # Check for LoRA checkpoints
    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
            
        if lora_path.exists():
            logger.info(f"Loading trained mPLUG-Owl LoRA weights from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            logger.warning(f"Could not find LoRA adapter at {args.out_checkpoint}. Using base model.")
            
    model = model.to(device)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    
    predictor = MplugOwlPredictor(model, processor, tokenizer, device)
    
    # Load dataset
    data_path = Path(args.jsonl_dir or config.dataset.data_path)
    if data_path.is_dir():
        target_path = data_path / "test.jsonl"
        if not target_path.exists():
            target_path = data_path / "dev.jsonl"
    else:
        target_path = data_path
        
    logger.info(f"Loading mPLUG-Owl test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    
    predictions = []
    
    logger.info("Starting mPLUG-Owl inference...")
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
    logger.info("--- mPLUG-Owl Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
