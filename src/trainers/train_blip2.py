"""
Training script specialized for BLIP-2 (Salesforce/blip2-opt-2.7b).

This module follows the repository training contract:
``run_train(args, config)`` receives paths and hyperparameters from the
dispatcher/config files, saves the best checkpoint by dev accuracy, and then
runs the matching inference pipeline on the test split.
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import (
    build_blip2_prompt,
    build_result_record,
    decode_blip2_output,
    resolve_split_paths,
)
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_json, load_jsonl
from ..utils.logging import setup_logger
from ..utils.seed import set_seed

logger = setup_logger(__name__)


class Blip2VQADataset(Dataset):
    """SpatialMQA dataset for BLIP-2 LoRA/full fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor: Blip2Processor,
        max_samples: int = None,
        max_label_length: int = 16,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_label_length = max_label_length

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
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        prompt = build_blip2_prompt(item["question"], item.get("options", []))
        answer = str(item["answer"])

        encoding = self.processor(
            images=image,
            text=prompt,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            answer,
            add_special_tokens=False,
            max_length=self.max_label_length - 1,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos = torch.tensor([eos_token_id], dtype=labels.dtype)
            labels = torch.cat([labels, eos], dim=0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": labels,
        }


class Blip2TrainCollator:
    """Pad BLIP-2 batches and mask label padding from the LM loss."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    @staticmethod
    def _pad_1d(tensors, padding_value: int):
        max_len = max(tensor.size(0) for tensor in tensors)
        padded = []
        for tensor in tensors:
            pad_size = max_len - tensor.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size,), padding_value, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=0)
            padded.append(tensor)
        return torch.stack(padded)

    def __call__(self, batch):
        return {
            "input_ids": self._pad_1d([item["input_ids"] for item in batch], self.pad_token_id),
            "attention_mask": self._pad_1d([item["attention_mask"] for item in batch], 0),
            "labels": self._pad_1d([item["labels"] for item in batch], self.label_pad_token_id),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _autocast_context(device, bf16: bool):
    dtype = torch.bfloat16 if bf16 else torch.float16
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    )


def compute_eval_loss(model, dataloader, device, bf16: bool = True) -> float:
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Valid]"):
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(device, bf16):
                outputs = model(**batch)
            eval_loss += outputs.loss.item()

    return eval_loss / max(len(dataloader), 1)


def compute_dev_accuracy(model, valid_dataset, processor, device, bf16: bool = True) -> float:
    """Compute dev accuracy using the same prompt/normalization as inference."""
    model.eval()
    predictions = []

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 1
        )

    with torch.no_grad():
        for idx in tqdm(range(len(valid_dataset)), desc="[Dev Accuracy]"):
            item = valid_dataset.data[idx]
            image_path = valid_dataset.image_dir / item["image"]
            image = Image.open(image_path).convert("RGB")
            prompt = build_blip2_prompt(item["question"], item.get("options", []))

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with _autocast_context(device, bf16):
                outputs = model.generate(**inputs, max_new_tokens=20)

            decoded = decode_blip2_output(processor, outputs[0], prompt)
            predictions.append(build_result_record(item, idx, decoded))

    metrics = calculate_spatial_metrics(predictions)
    return metrics["accuracy"]


def _build_model_and_processor(config: ExperimentConfig, device):
    logger.info("Building BLIP-2 model and processor...")
    processor = Blip2Processor.from_pretrained(config.model.model_name_or_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name_or_path,
        **kwargs,
    )

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 1
        )

    is_quantized = config.model.load_in_8bit or config.model.load_in_4bit
    if config.model.use_lora:
        if is_quantized:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            bias="none",
            target_modules=config.model.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if not is_quantized:
        model = model.to(device)

    return model, processor


def _save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)

    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint
    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _build_model_and_processor(config, device)

    data_path = args.jsonl_dir or config.dataset.data_path
    image_dir = args.image_dir or config.dataset.image_dir
    train_path, val_path = resolve_split_paths(data_path)

    logger.info(f"Loading BLIP-2 train dataset from {train_path}")
    train_dataset = Blip2VQADataset(
        data_path=str(train_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples,
    )
    logger.info(f"Loading BLIP-2 valid dataset from {val_path}")
    valid_dataset = Blip2VQADataset(
        data_path=str(val_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples,
    )

    batch_size = args.batch_size or config.training.batch_size
    collator = Blip2TrainCollator(
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 1
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and not config.training.bf16),
    )

    grad_accum_steps = max(
        1,
        getattr(config.training, "gradient_accumulation_steps", config.training.cal_num),
    )
    min_eval_loss = float("inf")
    best_dev_accuracy = float("-inf")
    early_stopping_hook = 0
    losses_history = []
    dev_loss_history = []
    dev_accuracy_history = []
    log_history = []
    global_step = 0

    logger.info("Starting BLIP-2 training loop...")
    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.training.num_epochs} [Train]")
        for idx, batch in enumerate(pbar):
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(device, config.training.bf16):
                outputs = model(**batch)
                loss = outputs.loss

            raw_loss = loss.item()
            epoch_loss += raw_loss
            scaler.scale(loss / grad_accum_steps).backward()

            should_step = (idx + 1) % grad_accum_steps == 0 or idx == len(train_dataloader) - 1
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": raw_loss})
            losses_history.append({"epoch": epoch + 1, "step": global_step, "loss": raw_loss})
            global_step += 1

        train_loss = epoch_loss / max(len(train_dataloader), 1)
        eval_loss = compute_eval_loss(model, valid_dataloader, device, bf16=config.training.bf16)
        min_eval_loss = min(min_eval_loss, eval_loss)
        dev_accuracy = compute_dev_accuracy(model, valid_dataset, processor, device, bf16=config.training.bf16)

        dev_loss_history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        dev_accuracy_history.append({"epoch": epoch + 1, "dev_accuracy": dev_accuracy})
        log_item = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "dev_accuracy": dev_accuracy,
            "best_dev_accuracy": max(best_dev_accuracy, dev_accuracy),
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_history.append(log_item)

        logger.info(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | Dev Acc: {dev_accuracy:.4f} | "
            f"LR: {log_item['lr']}"
        )
        scheduler.step()

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            early_stopping_hook = 0
            save_path = out_checkpoint / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"Saved best model with dev accuracy {dev_accuracy:.4f} to {save_path}")
        else:
            early_stopping_hook += 1
            if early_stopping_hook > config.training.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        _save_json(out_results / "losses.json", losses_history)
        _save_json(out_results / "dev_loss.json", dev_loss_history)
        _save_json(out_results / "dev_accuracy.json", dev_accuracy_history)
        _save_json(out_results / "log.json", log_history)

    _save_json(
        out_results / "best_dev_metric.json",
        {"best_dev_accuracy": best_dev_accuracy, "best_dev_loss": min_eval_loss},
    )

    logger.info("BLIP-2 training complete. Loading best model for evaluation on test set...")
    best_model_path = out_checkpoint / "best_model"
    if best_model_path.exists():
        from ..inference.inference_blip2 import run_infer as blip2_infer

        class Args:
            pass

        infer_args = Args()
        infer_args.out_checkpoint = str(best_model_path)
        infer_args.out_results = str(out_results)
        infer_args.jsonl_dir = args.jsonl_dir or config.dataset.data_path
        infer_args.image_dir = args.image_dir or config.dataset.image_dir
        infer_args.batch_size = args.batch_size or config.training.batch_size

        blip2_infer(infer_args, config)
    else:
        logger.warning("Best model not found; skipping test evaluation.")
