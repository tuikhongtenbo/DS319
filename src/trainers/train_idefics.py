"""
Training script specialized for Idefics (HuggingFaceM4/idefics-9b-instruct).
"""

import logging
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, IdeficsForVisionText2Text, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_json, load_jsonl
from ..utils.seed import set_seed
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class IdeficsDataset(Dataset):
    """Dataset class for Idefics training."""
    def __init__(self, data_path: str, image_dir: str, max_samples: int = None):
        self.image_dir = Path(image_dir)
        
        path = Path(data_path)
        if path.suffix == ".jsonl":
            self.data = load_jsonl(path)
        else:
            self.data = load_json(path)
            
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        question = item['question']
        answer = str(item['answer'])
        options = item.get("options", [])
        image_id = item['image']
        image_path = self.image_dir / image_id
        
        image = Image.open(image_path).convert("RGB")
        
        options_str = "; ".join(options)
        
        # Structure the prompt for Idefics instruct
        # User prompt contains image and question
        # Assistant responds with answer
        prompt = [
            image,
            f"User: Question: {question} Options: {options_str}\nAssistant: {answer}<end_of_utterance>"
        ]
        
        return prompt

class IdeficsCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # batch is a list of prompts (each prompt is a list [image, text])
        inputs = self.processor(batch, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

@torch.no_grad()
def evaluate_idefics(model, dataloader, processor, device, bf16=True):
    model.eval()
    predictions = []
    
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids if len(bad_words) > 0 else None
    
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    
    for batch_items in tqdm(dataloader, desc="[Valid]"):
        # For validation generation, we only pass prompt up to "Assistant:"
        eval_prompts = []
        answers = []
        for prompt in batch_items:
            # Reconstruct without answer for generation
            # prompt is a list: [image, text_with_answer]
            image = prompt[0]
            text_with_answer = prompt[1]
            
            # Extract prompt and target answer
            prompt_text = text_with_answer.split("Assistant:")[0] + "Assistant:"
            answer_text = text_with_answer.split("Assistant:")[-1].replace("<end_of_utterance>", "").strip()
            
            eval_prompts.append([image, prompt_text])
            answers.append(answer_text)
            
        inputs = processor(eval_prompts, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                eos_token_id=[eos_token_id],
                bad_words_ids=bad_words_ids,
                max_new_tokens=20
            )
            
        decoded_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i in range(len(answers)):
            # Idefics generate output includes the prompt, extract the assistant's answer
            output_text = decoded_outputs[i]
            if "Assistant:" in output_text:
                output_text = output_text.split("Assistant:")[-1].strip()
            else:
                output_text = output_text.strip()
                
            predictions.append({
                "output": output_text.lower(),
                "answer": answers[i].lower()
            })
            
    return calculate_spatial_metrics(predictions)

def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)
    
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint
    
    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building Idefics model and processor...")
    
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
    
    if config.model.use_lora:
        logger.info("Wrapping Idefics model with LoRA...")
        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            bias="none",
            target_modules=config.model.lora_target_modules
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.model.load_in_8bit and not config.model.load_in_4bit:
        model = model.to(device)
        
    # Setup Datasets
    data_path = Path(args.jsonl_dir or config.dataset.data_path)
    image_dir = args.image_dir or config.dataset.image_dir
    
    if data_path.is_dir():
        train_path = data_path / "train.jsonl"
        val_path = data_path / "dev.jsonl"
        if not val_path.exists():
            val_path = train_path
    else:
        train_path = data_path
        val_path = data_path
        
    logger.info(f"Loading Idefics train dataset from {train_path}")
    train_dataset = IdeficsDataset(data_path=str(train_path), image_dir=image_dir, max_samples=config.dataset.max_samples)
    
    logger.info(f"Loading Idefics valid dataset from {val_path}")
    valid_dataset = IdeficsDataset(data_path=str(val_path), image_dir=image_dir, max_samples=config.dataset.max_samples)
    
    collator = IdeficsCollator(processor)
    batch_size = args.batch_size or config.training.batch_size
    
    # Validation Dataloader uses batch_size=1 and custom collation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=False)
    # For evaluation, we pass raw items to generate the prompt text and parse it
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x, pin_memory=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.bf16)
    
    best_eval_acc = -1.0
    early_stopping_hook = 0
    
    losses_history = []
    dev_acc_history = []
    log_history = []
    
    global_step = 0
    
    logger.info("Starting Idefics training loop...")
    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        cal_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Train]")
        for idx, batch in enumerate(pbar):
            # Move inputs to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            
            dtype = torch.bfloat16 if config.training.bf16 else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
                
            epoch_loss += loss.item()
            cal_loss += loss
            
            if (idx + 1) % config.training.cal_num == 0 or idx == len(train_dataloader) - 1:
                divisor = config.training.cal_num if (idx + 1) % config.training.cal_num == 0 else ((idx + 1) % config.training.cal_num)
                cal_loss = cal_loss / divisor
                
                scaler.scale(cal_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                cal_loss = 0.0
                
            pbar.set_postfix({"loss": loss.item()})
            losses_history.append({"epoch": epoch+1, "step": global_step, "loss": loss.item()})
            global_step += 1
            
        # Validation
        eval_metrics = evaluate_idefics(model, valid_dataloader, processor, device, bf16=config.training.bf16)
        eval_acc = eval_metrics["accuracy"]
        dev_acc_history.append({"epoch": epoch+1, "dev_acc": eval_acc})
        
        log_item = {
            "epoch": epoch+1,
            "train_loss": epoch_loss / len(train_dataloader),
            "eval_acc": eval_acc,
            "lr": optimizer.param_groups[0]['lr']
        }
        log_history.append(log_item)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {log_item['train_loss']:.4f} | Eval Acc: {eval_acc:.4f} | LR: {log_item['lr']}")
        scheduler.step()
        
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            early_stopping_hook = 0
            
            # Save checkpoint overriding previous best
            save_path = out_checkpoint / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"Saved best model with acc {eval_acc:.4f} to {save_path}")
        else:
            early_stopping_hook += 1
            if early_stopping_hook > config.training.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        # Save tracking files each epoch
        with open(out_results / "losses.json", "w", encoding="utf-8") as f:
            json.dump(losses_history, f, indent=4)
        with open(out_results / "dev_acc.json", "w", encoding="utf-8") as f:
            json.dump(dev_acc_history, f, indent=4)
        with open(out_results / "log.json", "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=4)
            
    logger.info("Idefics training complete.")
