"""
Training script specialized for InstructBLIP (Salesforce/instructblip-flan-t5-xl).
"""

import logging
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

from ..configs.config import ExperimentConfig
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_json, load_jsonl
from ..utils.seed import set_seed
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class InstructBLIPDataset(Dataset):
    """Dataset class for InstructBLIP training."""
    def __init__(self, data_path: str, image_dir: str, processor: InstructBlipProcessor, max_samples: int = None):
        self.image_dir = Path(image_dir)
        self.processor = processor
        
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
        image_id = item['image']
        image_path = self.image_dir / image_id
        image = Image.open(image_path).convert("RGB")

        # Process image and question
        encoding = self.processor(images=image, text=question, return_tensors="pt")
        
        # Tokenize answer as labels
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, padding="max_length", truncation=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
            
        return encoding

@torch.no_grad()
def evaluate_instructblip(model, dataloader, processor, device, bf16=True):
    model.eval()
    predictions = []
    
    dtype = torch.bfloat16 if bf16 else torch.float16
    for batch in tqdm(dataloader, desc="[Valid]"):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        labels = batch.pop("labels").to(device)
        
        # instructblip processor generates qformer_input_ids and mask
        qformer_input_ids = batch.pop("qformer_input_ids", None)
        qformer_attention_mask = batch.pop("qformer_attention_mask", None)
        
        gen_kwargs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "max_new_tokens": 20
        }
        if qformer_input_ids is not None:
            gen_kwargs["qformer_input_ids"] = qformer_input_ids.to(device)
        if qformer_attention_mask is not None:
            gen_kwargs["qformer_attention_mask"] = qformer_attention_mask.to(device)
            
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            outputs = model.generate(**gen_kwargs)
            
        # Decode answers
        decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Decode targets (ignore padding token 0/1/pad_token_id)
        pad_token_id = processor.tokenizer.pad_token_id or 0
        for i in range(labels.size(0)):
            valid_label_tokens = labels[i][labels[i] != pad_token_id]
            decoded_answer = processor.tokenizer.decode(valid_label_tokens, skip_special_tokens=True).strip()
            output_text = decoded_outputs[i].strip()
            
            predictions.append({
                "output": output_text.lower(),
                "answer": decoded_answer.lower()
            })
            
    return calculate_spatial_metrics(predictions)

def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)
    
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint
    
    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building InstructBLIP model and processor...")
    processor = InstructBlipProcessor.from_pretrained(config.model.model_name_or_path)
    
    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif config.model.load_in_4bit:
        kwargs["load_in_4bit"] = True
        
    model = InstructBlipForConditionalGeneration.from_pretrained(config.model.model_name_or_path, **kwargs)
    
    if config.model.use_lora:
        logger.info("Wrapping model with LoRA...")
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
        
    logger.info(f"Loading InstructBLIP train dataset from {train_path}")
    train_dataset = InstructBLIPDataset(
        data_path=str(train_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples
    )
    
    logger.info(f"Loading InstructBLIP valid dataset from {val_path}")
    valid_dataset = InstructBLIPDataset(
        data_path=str(val_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples
    )
    
    batch_size = args.batch_size or config.training.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.bf16)
    
    best_eval_acc = -1.0
    early_stopping_hook = 0
    
    losses_history = []
    dev_acc_history = []
    log_history = []
    
    global_step = 0
    
    logger.info("Starting InstructBLIP training loop...")
    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        cal_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Train]")
        for idx, batch in enumerate(pbar):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            labels = batch.pop("labels").to(device)
            
            # instructblip specific inputs
            qformer_input_ids = batch.pop("qformer_input_ids", None)
            qformer_attention_mask = batch.pop("qformer_attention_mask", None)
            
            optimizer.zero_grad()
            
            dtype = torch.bfloat16 if config.training.bf16 else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    qformer_input_ids=qformer_input_ids.to(device) if qformer_input_ids is not None else None,
                    qformer_attention_mask=qformer_attention_mask.to(device) if qformer_attention_mask is not None else None,
                    labels=labels
                )
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
        eval_metrics = evaluate_instructblip(model, valid_dataloader, processor, device, bf16=config.training.bf16)
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
            
    logger.info("InstructBLIP training complete.")
