"""
Inference script specialized for LLaVA (liuhaotian/llava-v1.5-7b).
"""

import os
import logging
from pathlib import Path
import json
import torch
import re
from tqdm import tqdm
from PIL import Image

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

def run_infer(args, config: ExperimentConfig):
    # Dynamic imports since llava libraries might not be installed on Windows development environments
    try:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    except ImportError as e:
        logger.error(f"Failed to import llava libraries. Ensure LLaVA is installed: {e}")
        logger.warning("Skipping LLaVA inference run.")
        return
        
    disable_torch_init()
    
    model_path = config.model.model_name_or_path
    model_name = get_model_name_from_path(model_path)
    
    logger.info(f"Loading base LLaVA model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name=model_name
    )
    
    # Load LoRA checkpoint if exists
    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        # Check for both best_model and saved_model (HF Trainer outputs)
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
            
        if lora_path.exists():
            logger.info(f"Loading trained LLaVA LoRA weights from {lora_path}")
            model.load_adapter(str(lora_path))
        else:
            logger.warning(f"Could not find LoRA adapter at {args.out_checkpoint}. Using base model.")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load dataset
    data_path = Path(args.jsonl_dir or config.dataset.data_path)
    if data_path.is_dir():
        target_path = data_path / "test.jsonl"
        if not target_path.exists():
            target_path = data_path / "dev.jsonl"
    else:
        target_path = data_path
        
    logger.info(f"Loading LLaVA test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)
    
    predictions = []
    
    # Determine conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"
        
    logger.info(f"Using LLaVA conversation template: {conv_mode}")
    
    logger.info("Starting LLaVA inference...")
    for item in tqdm(test_data):
        image_path = image_dir / item["image"]
        question = item["question"]
        options = item.get("options", [])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            continue
            
        # Format prompt
        options_str = "; ".join(options)
        prompt_text = f"You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question, and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: , Question: {question}, Options: {options_str}. \n Output:"
        
        qs = prompt_text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        images_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
        image_sizes = [image.size]
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=20,
                use_cache=True
            )
            
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
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
    logger.info("--- LLaVA Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
