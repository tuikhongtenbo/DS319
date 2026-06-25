"""
Standard training loop for BLIP/BLIP2 models.
"""

import logging
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.config import TrainingConfig
from ..metrics.metrics import calculate_spatial_metrics

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        processor: any,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        out_checkpoint: Path,
        out_results: Path
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        self.device = device
        self.out_checkpoint = Path(out_checkpoint)
        self.out_results = Path(out_results)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1) # Original code uses ignore_index=1, often padding token or similar
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.bf16)
        
        self.out_results.mkdir(parents=True, exist_ok=True)
        self.out_checkpoint.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        logger.info("Starting training loop...")
        
        best_eval_acc = -1.0
        early_stopping_hook = 0
        
        losses_history = []
        dev_acc_history = []
        log_history = []
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            cal_loss = 0.0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]")
            for idx, batch in enumerate(pbar):
                # Move to device
                input_ids = batch.pop("input_ids").to(self.device)
                pixel_values = batch.pop("pixel_values").to(self.device)
                attention_mask = batch.pop("attention_mask").to(self.device)
                labels = batch.pop("labels").to(self.device)
                
                # Autocast
                dtype = torch.bfloat16 if self.config.bf16 else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                
                # Calculate loss
                # Shape matching based on original code
                loss = self.criterion(
                    logits.view(-1, logits.shape[-1])[:labels.shape[1], :].contiguous(),
                    labels.view(-1).contiguous()
                )
                
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                
                cal_loss += loss
                if (idx + 1) % self.config.cal_num == 0 or idx == len(self.train_dataloader) - 1:
                    divisor = self.config.cal_num if (idx + 1) % self.config.cal_num == 0 else ((idx + 1) % self.config.cal_num)
                    cal_loss = cal_loss / divisor
                    
                    self.scaler.scale(cal_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    cal_loss = 0.0
                    
                pbar.set_postfix({"loss": loss.item()})
                
                losses_history.append({"epoch": epoch+1, "step": global_step, "loss": loss.item()})
                global_step += 1
                
            # Validation
            eval_metrics = self.evaluate()
            eval_acc = eval_metrics["accuracy"]
            
            dev_acc_history.append({"epoch": epoch+1, "dev_acc": eval_acc})
            
            log_item = {
                "epoch": epoch+1, 
                "train_loss": epoch_loss/len(self.train_dataloader), 
                "eval_acc": eval_acc,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            log_history.append(log_item)
            
            logger.info(f"Epoch {epoch+1} | Train Loss: {log_item['train_loss']:.4f} | Eval Acc: {eval_acc:.4f} | LR: {log_item['lr']}")
            self.scheduler.step()
            
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                early_stopping_hook = 0
                
                # Save checkpoint overriding previous best
                save_path = self.out_checkpoint / "best_model"
                self.model.save_pretrained(save_path)
                logger.info(f"Saved best model with acc {eval_acc:.4f} to {save_path}")
            else:
                early_stopping_hook += 1
                if early_stopping_hook > self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
            # Save tracking files each epoch
            self._save_json(losses_history, self.out_results / "losses.json")
            self._save_json(dev_acc_history, self.out_results / "dev_acc.json")
            self._save_json(log_history, self.out_results / "log.json")
                    
        logger.info("Training complete.")

    def _save_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        pbar = tqdm(self.valid_dataloader, desc="[Valid]")
        
        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        predictions = []
        
        for batch in pbar:
            input_ids = batch.pop("input_ids").to(self.device)
            pixel_values = batch.pop("pixel_values").to(self.device)
            attention_mask = batch.pop("attention_mask").to(self.device)
            labels = batch.pop("labels").to(self.device) # Expected token IDs
            
            # For accuracy calculation, we should generate text and compare with decoded labels.
            # However, during standard generation we just pass the prompt. 
            # In validation, since we don't have direct access to raw text without decoding,
            # we will decode the labels and generated outputs.
            
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=20
                )
            
            # Decode outputs
            decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Decode labels (ignore pad token -100 or 1 if used)
            # Find indices where label is not 1 (the ignore index used above)
            for i in range(labels.size(0)):
                valid_label_tokens = labels[i][labels[i] != 1]
                decoded_answer = self.processor.decode(valid_label_tokens, skip_special_tokens=True).strip()
                
                # Predictor formatting
                output_text = decoded_outputs[i].strip()
                
                predictions.append({
                    "output": output_text,
                    "answer": decoded_answer
                })
            
        metrics = calculate_spatial_metrics(predictions)
        return metrics
