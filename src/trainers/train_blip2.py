"""
Training script specialized for BLIP-2 (Salesforce/blip2-opt-2.7b).
Selects best model based on dev accuracy, not loss.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

from ..configs.config import ExperimentConfig
from ..datasets.collator import Blip2Collator
from ..datasets.preprocessing import build_blip2_prompt, build_result_record, resolve_split_paths
from ..metrics.metrics import calculate_spatial_metrics
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
        options = item.get("options", [])
        prompt = build_blip2_prompt(question, options)
        answer = str(item["answer"])
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(images=image, text=prompt, return_tensors="pt")
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
    bf16: bool = True,
) -> torch.Tensor:
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_mask = batch.pop("attention_mask").to(device)
    labels = batch.pop("labels").to(device)

    dtype = torch.bfloat16 if bf16 else torch.float16
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    return outputs.loss


def compute_eval_loss(
    model,
    dataloader,
    device,
    bf16: bool = True,
) -> float:
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Valid]"):
            batch_copy = {key: value.clone() for key, value in batch.items()}
            loss = compute_blip2_loss(model, batch_copy, device, bf16=bf16)
            eval_loss += loss.item()

    return eval_loss / len(dataloader)


def compute_dev_accuracy(
    model,
    valid_dataset,
    processor,
    device,
    bf16: bool = True,
    batch_size: int = 16,
) -> float:
    """Compute accuracy on dev set for model selection."""
    model.eval()
    predictions = []
    raw_data = getattr(valid_dataset, "data", valid_dataset)

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id or 1

    try:
        with torch.inference_mode():
            index_batches = range(0, len(valid_dataset), batch_size)
            for start_idx in tqdm(index_batches, desc="[Dev Accuracy]"):
                batch_indices = list(range(start_idx, min(start_idx + batch_size, len(valid_dataset))))
                batch_items = [raw_data[idx] for idx in batch_indices]
                images = []
                for item in batch_items:
                    with Image.open(valid_dataset.image_dir / item["image"]) as image:
                        images.append(image.convert("RGB"))
                prompts = [
                    build_blip2_prompt(item["question"], item.get("options", []))
                    for item in batch_items
                ]
                inputs = processor(
                    images=images,
                    text=prompts,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                outputs = model.generate(**inputs, max_new_tokens=20)
                decoded_answers = processor.batch_decode(outputs, skip_special_tokens=True)

                for idx, item, prompt, decoded in zip(batch_indices, batch_items, prompts, decoded_answers):
                    pred_answer = decoded.strip()
                    if pred_answer.lower().startswith(prompt.lower()):
                        pred_answer = pred_answer[len(prompt):].strip()
                    pred_answer = pred_answer.rstrip(".")
                    if not pred_answer:
                        pred_answer = "--"
                    predictions.append(build_result_record(item, idx, pred_answer))
    finally:
        processor.tokenizer.padding_side = original_padding_side

    metrics = calculate_spatial_metrics(predictions)
    return metrics["accuracy"]


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
        label_pad_token_id=-100,
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

    min_eval_loss = float("inf")
    best_dev_accuracy = 0.0
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
        cal_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.training.num_epochs} [Train]")
        for idx, batch in enumerate(pbar):
            batch_copy = {key: value.clone() for key, value in batch.items()}
            loss = compute_blip2_loss(
                model, batch_copy, device, bf16=config.training.bf16
            )

            epoch_loss += loss.item()
            cal_loss += loss

            if (idx + 1) % config.training.cal_num == 0 or idx == len(train_dataloader) - 1:
                if (idx + 1) % config.training.cal_num == 0:
                    divisor = config.training.cal_num
                else:
                    divisor = (idx + 1) % config.training.cal_num
                averaged_loss = cal_loss / divisor
                optimizer.zero_grad()
                scaler.scale(averaged_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                cal_loss = 0.0

            pbar.set_postfix({"loss": loss.item()})
            losses_history.append({"epoch": epoch + 1, "step": global_step, "loss": loss.item()})
            global_step += 1

        eval_loss = compute_eval_loss(
            model, valid_dataloader, device, bf16=config.training.bf16
        )
        dev_loss_history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        min_eval_loss = min(min_eval_loss, eval_loss)

        dev_accuracy = compute_dev_accuracy(
            model, valid_dataset, processor, device,
            bf16=config.training.bf16, batch_size=batch_size
        )
        dev_accuracy_history.append({"epoch": epoch + 1, "dev_accuracy": dev_accuracy})

        log_item = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_dataloader),
            "eval_loss": eval_loss,
            "dev_accuracy": dev_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_history.append(log_item)

        logger.info(
            f"Epoch {epoch + 1} | Train Loss: {log_item['train_loss']:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | Dev Acc: {dev_accuracy:.4f} | LR: {log_item['lr']}"
        )
        scheduler.step()

        # Select best model based on dev accuracy, not loss
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            early_stopping_hook = 0
            save_path = out_checkpoint / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"Saved best model with dev accuracy {dev_accuracy:.4f} to {save_path}")
        else:
            early_stopping_hook += 1
            if early_stopping_hook > config.training.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        with open(out_results / "losses.json", "w", encoding="utf-8") as file:
            json.dump(losses_history, file, indent=4)
        with open(out_results / "dev_loss.json", "w", encoding="utf-8") as file:
            json.dump(dev_loss_history, file, indent=4)
        with open(out_results / "dev_accuracy.json", "w", encoding="utf-8") as file:
            json.dump(dev_accuracy_history, file, indent=4)
        with open(out_results / "log.json", "w", encoding="utf-8") as file:
            json.dump(log_history, file, indent=4)

    # Save final best metric
    with open(out_results / "best_dev_metric.json", "w", encoding="utf-8") as file:
        json.dump({"best_dev_accuracy": best_dev_accuracy, "best_dev_loss": min_eval_loss}, file, indent=4)

    logger.info("BLIP-2 training complete. Loading best model for evaluation on test set...")

    best_model_path = out_checkpoint / "best_model"
    if best_model_path.exists():
        from ..inference.inference_blip2 import run_infer as blip2_infer

        class Args:
            pass

        infer_args = Args()
        infer_args.out_checkpoint = str(out_checkpoint)
        infer_args.out_results = str(out_results)
        infer_args.jsonl_dir = args.jsonl_dir or config.dataset.data_path
        infer_args.image_dir = args.image_dir or config.dataset.image_dir
        infer_args.batch_size = args.batch_size or config.training.batch_size

        blip2_infer(infer_args, config)
    else:
        logger.warning("Best model not found; skipping test evaluation.")
