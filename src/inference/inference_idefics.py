"""
Inference script specialized for Idefics (HuggingFaceM4/idefics-9b-instruct).
"""

import logging
from pathlib import Path
import json
import torch
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, IdeficsForVisionText2Text, BitsAndBytesConfig
from peft import PeftModel

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class IdeficsPredictor:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
        
        tokenizer = processor.tokenizer
        bad_words = ["<image>", "<fake_token_around_image>"]
        self.bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids if len(bad_words) > 0 else None
        
        eos_token = "</s>"
        self.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: list) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"
            
        options_str = "; ".join(options)
        
        # Structure the prompt for Idefics instruct
        prompt = [
            image,
            f"User: Question: {question} Options: {options_str}\nAssistant:"
        ]
        
        inputs = self.processor([prompt], return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                eos_token_id=[self.eos_token_id],
                bad_words_ids=self.bad_words_ids,
                max_new_tokens=20
            )
            
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Extract assistant output
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1].strip()
        return decoded

def run_infer(args, config: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Building Idefics model and processor for inference...")
    
    bnb_config = None
    if config.model.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "embed_tokens"]
        )
    elif config.model.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"]
        )
        
    processor = AutoProcessor.from_pretrained(config.model.model_name_or_path)
    
    kwargs = {"device_map": config.model.device_map}
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
        
    model = IdeficsForVisionText2Text.from_pretrained(config.model.model_name_or_path, **kwargs)
    
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
        
    predictor = IdeficsPredictor(model, processor, device)
    
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
    
    logger.info("Starting Idefics inference...")
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
    logger.info("--- Idefics Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
