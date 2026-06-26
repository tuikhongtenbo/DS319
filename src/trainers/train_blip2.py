"""
Training script specialized for BLIP-2 (Salesforce/blip2-opt-2.7b).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

from ..configs.config import ExperimentConfig
from ..datasets.collator import Blip2Collator
from ..datasets.preprocessing import resolve_split_paths
from ..utils.io import load_json, load_jsonl
from ..utils.logging import setup_logger
from ..utils.seed import set_seed

logger = setup_logger(__name__)


class ImageCaptioningDataset(Dataset):
    """Dataset class for BLIP-2 LoRA training."""

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor: Blip2Processor,
        max_samples: int = None,
    ):
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
        question = item["question"]
        answer = str(item["answer"])
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(images=image, text=question, return_tensors="pt")
        labels = self.processor.tokenizer(
            answer, return_tensors="pt", add_special_tokens=False
        ).input_ids
        eos_token_id = self.processor.tokenizer.eos_token_id or 50118
        labels = torch.cat((labels, torch.tensor([[eos_token_id]])), dim=1)
        encoding["labels"] = labels

        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)

        return encoding


def compute_blip2_loss(
    model,
    batch: dict,
    device: torch.device,
    criterion: nn.Module,
    bf16: bool = True,
) -> torch.Tensor:
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_mask = batch.pop("attention_mask").to(device)
    labels = batch.pop("labels").to(device)

    dtype = torch.bfloat16 if bf16 else torch.float16
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        ).logits

    # Logits seq-len matches input_ids (encoder output), not labels (decoder target).
    # Only the last |labels| positions correspond to the decoder targets.
    # Slice logits accordingly so shapes align for cross_entropy.
    label_len = labels.shape[1]
    logits_sliced = logits[:, -label_len:, :].contiguous()

    return criterion(
        logits_sliced.view(-1, logits_sliced.shape[-1]).contiguous(),
        labels.view(-1).contiguous(),
    )


def compute_eval_loss(
    model,
    dataloader,
    device,
    criterion,
    bf16: bool = True,
) -> float:
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Valid]"):
            batch_copy = {key: value.clone() for key, value in batch.items()}
            loss = compute_blip2_loss(model, batch_copy, device, criterion, bf16=bf16)
            eval_loss += loss.item()

    return eval_loss / len(dataloader)


def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)

    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    logger.info("Building BLIP-2 model and processor...")
    processor = Blip2Processor.from_pretrained(config.model.model_name_or_path)

    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name_or_path, **kwargs
    )

    if config.model.use_lora:
        logger.info("Wrapping model with LoRA...")
        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            bias="none",
            target_modules=config.model.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.model.load_in_8bit and not config.model.load_in_4bit:
        model = model.to(device)

    data_path = args.jsonl_dir or config.dataset.data_path
    image_dir = args.image_dir or config.dataset.image_dir
    train_path, val_path = resolve_split_paths(data_path)

    logger.info(f"Loading BLIP-2 train dataset from {train_path}")
    train_dataset = ImageCaptioningDataset(
        data_path=str(train_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples,
    )

    logger.info(f"Loading BLIP-2 valid dataset from {val_path}")
    valid_dataset = ImageCaptioningDataset(
        data_path=str(val_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples,
    )

    batch_size = args.batch_size or config.training.batch_size
    collator = Blip2Collator(
        pad_token_id=processor.tokenizer.pad_token_id or 1,
        label_pad_token_id=1,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collator,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.bf16)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    min_eval_loss = float("inf")
    early_stopping_hook = 0
    losses_history = []
    dev_loss_history = []
    log_history = []
    global_step = 0

    logger.info("Starting BLIP-2 training loop...")
    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        cal_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.training.num_epochs} [Train]")
        for idx, batch in enumerate(pbar):
            batch_copy = {key: value.clone() for key, value in batch.items()}
            loss = compute_blip2_loss(
                model, batch_copy, device, criterion, bf16=config.training.bf16
            )

            epoch_loss += loss.item()
            cal_loss += loss

            if (idx + 1) % config.training.cal_num == 0 or idx == len(train_dataloader) - 1:
                averaged_loss = cal_loss / config.training.cal_num
                optimizer.zero_grad()
                scaler.scale(averaged_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                cal_loss = 0.0

            pbar.set_postfix({"loss": loss.item()})
            losses_history.append({"epoch": epoch + 1, "step": global_step, "loss": loss.item()})
            global_step += 1

        eval_loss = compute_eval_loss(
            model, valid_dataloader, device, criterion, bf16=config.training.bf16
        )
        dev_loss_history.append({"epoch": epoch + 1, "eval_loss": eval_loss})

        log_item = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_dataloader),
            "eval_loss": eval_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_history.append(log_item)

        logger.info(
            f"Epoch {epoch + 1} | Train Loss: {log_item['train_loss']:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | LR: {log_item['lr']}"
        )
        scheduler.step()

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            early_stopping_hook = 0
            save_path = out_checkpoint / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"Saved best model with eval loss {eval_loss:.4f} to {save_path}")
        else:
            early_stopping_hook += 1
            if early_stopping_hook > config.training.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        with open(out_results / "losses.json", "w", encoding="utf-8") as file:
            json.dump(losses_history, file, indent=4)
        with open(out_results / "dev_loss.json", "w", encoding="utf-8") as file:
            json.dump(dev_loss_history, file, indent=4)
        with open(out_results / "log.json", "w", encoding="utf-8") as file:
            json.dump(log_history, file, indent=4)

    # Save final best metric
    with open(out_results / "last_dev_metric.json", "w", encoding="utf-8") as file:
        json.dump({"best_eval_loss": min_eval_loss}, file, indent=4)

    logger.info("BLIP-2 training complete. Running evaluation on test set...")

    best_model_path = out_checkpoint / "best_model"
    if best_model_path.exists():
        from src.inference.inference_blip2 import run_infer as blip2_infer

        class Args:
            out_checkpoint = str(best_model_path)
            out_results = str(out_results)
            jsonl_dir = args.jsonl_dir or config.dataset.data_path
            image_dir = args.image_dir or config.dataset.image_dir
            batch_size = args.batch_size or config.training.batch_size

        blip2_infer(Args(), config)
    else:
        logger.warning("Best model not found; skipping test evaluation.")
