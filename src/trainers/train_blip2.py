import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
)

from ..configs.config import ExperimentConfig
from ..datasets.collator import Blip2Collator
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


# ============================================================
# Utility functions
# ============================================================

def normalize_answer_text(answer: Any) -> str:
    """
    Normalize answer for safer comparison/debugging.

    Examples:
    - " A. Left " -> "a left"
    - "Left." -> "left"
    - 0 -> "0"
    """
    text = str(answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(".", "")
    text = text.replace(")", "")
    text = text.replace(":", "")
    return text.strip()


def strip_option_prefix(option: Any) -> str:
    """
    Remove option prefix such as:
    - "A. left" -> "left"
    - "B) right" -> "right"
    - "C: above" -> "above"
    """
    text = str(option).strip()
    text = re.sub(r"^[A-Da-d][\.\)\:]\s*", "", text)
    return text.strip()


def is_letter_answer(answer: Any) -> bool:
    return str(answer).strip().upper() in ["A", "B", "C", "D"]


def is_digit_answer(answer: Any) -> bool:
    return str(answer).strip().isdigit()


def build_option_scoring_targets(
    options: List[Any],
    gold_answer: Any,
) -> Tuple[List[str], str]:
    """
    Decide what strings should be scored by BLIP-2 during dev accuracy.

    Important:
    Training uses:
        answer = str(item["answer"])

    So during option scoring, the target strings should match the format
    of item["answer"], not necessarily the full option text.

    Cases:
    1. gold = "A"/"B"/"C"/"D"
       -> score ["A", "B", "C", "D"]

    2. gold = 0/1/2/3 or "0"/"1"/"2"/"3"
       -> score ["0", "1", "2", "3"]

    3. gold is full answer text, e.g. "left"
       -> score normalized option text, e.g. ["left", "right", ...]

    4. fallback:
       -> score raw option text
    """
    gold_str = str(gold_answer).strip()

    if is_letter_answer(gold_answer):
        letters = ["A", "B", "C", "D"]
        return letters[:len(options)], "letter"

    if is_digit_answer(gold_answer):
        indices = [str(i) for i in range(len(options))]
        return indices, "index"

    stripped_options = [strip_option_prefix(option) for option in options]

    # If gold matches stripped option content, score stripped option content.
    normalized_gold = normalize_answer_text(gold_answer)
    normalized_stripped = [normalize_answer_text(option) for option in stripped_options]

    if normalized_gold in normalized_stripped:
        return stripped_options, "option_text"

    # Otherwise use raw options.
    return [str(option) for option in options], "raw_option"


def convert_best_index_to_prediction(
    best_idx: int,
    options: List[Any],
    gold_answer: Any,
    mode: str,
) -> Any:
    """
    Convert best option index into prediction format expected by metric.

    The goal is: pred_answer should have the same format as item["answer"].
    """
    if mode == "letter":
        return ["A", "B", "C", "D"][best_idx]

    if mode == "index":
        # Preserve gold type if possible.
        if isinstance(gold_answer, int):
            return best_idx
        return str(best_idx)

    if mode == "option_text":
        return strip_option_prefix(options[best_idx])

    return str(options[best_idx])


# ============================================================
# Dataset
# ============================================================

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        question = item["question"]
        options = item.get("options", [])
        prompt = build_blip2_prompt(question, options)

        # Important: this is the target format used during training.
        answer = str(item["answer"])

        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = 50118

        eos_tensor = torch.tensor([[eos_token_id]], dtype=labels.dtype)
        labels = torch.cat((labels, eos_tensor), dim=1)

        encoding["labels"] = labels

        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)

        return encoding


# ============================================================
# Loss
# ============================================================

def compute_blip2_loss(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    bf16: bool = True,
) -> torch.Tensor:
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_mask = batch.pop("attention_mask").to(device)
    labels = batch.pop("labels").to(device)

    dtype = torch.bfloat16 if bf16 else torch.float16
    autocast_enabled = device.type == "cuda"

    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
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
    device: torch.device,
    bf16: bool = True,
) -> float:
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Valid Loss]"):
            batch_copy = {
                key: value.clone()
                for key, value in batch.items()
            }
            loss = compute_blip2_loss(
                model,
                batch_copy,
                device,
                bf16=bf16,
            )
            eval_loss += loss.item()

    return eval_loss / max(len(dataloader), 1)


# ============================================================
# Dev accuracy
# ============================================================

def _score_blip2_answer(
    model,
    processor: Blip2Processor,
    image: Image.Image,
    prompt: str,
    answer: str,
    device: torch.device,
    bf16: bool = True,
) -> float:
    """
    Score one candidate answer by teacher-forcing loss.
    Lower loss means the model thinks this answer is more likely.
    """
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    labels = processor.tokenizer(
        str(answer),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = 50118

    eos_tensor = torch.tensor([[eos_token_id]], dtype=labels.dtype)
    labels = torch.cat((labels, eos_tensor), dim=1).to(device)

    dtype = torch.bfloat16 if bf16 else torch.float16
    autocast_enabled = device.type == "cuda"

    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

    return float(outputs.loss.item())


def compute_dev_accuracy(
    model,
    valid_dataset,
    processor: Blip2Processor,
    device: torch.device,
    bf16: bool = True,
    batch_size: int = 16,
) -> float:
    """
    Compute multiple-choice dev accuracy.

    Fixed logic:
    - If answer is A/B/C/D, score A/B/C/D.
    - If answer is 0/1/2/3, score 0/1/2/3.
    - If answer is text, score option text.
    - Prediction is converted back to the same format as gold answer.
    """
    model.eval()
    predictions = []
    debug_examples = []

    raw_data = getattr(valid_dataset, "data", valid_dataset)

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id or 1

    try:
        with torch.inference_mode():
            index_batches = range(0, len(valid_dataset), batch_size)

            for start_idx in tqdm(index_batches, desc="[Dev Accuracy]"):
                end_idx = min(start_idx + batch_size, len(valid_dataset))
                batch_indices = list(range(start_idx, end_idx))

                for idx in batch_indices:
                    item = raw_data[idx]

                    question = item["question"]
                    options = item.get("options", [])
                    gold_answer = item.get("answer")

                    prompt = build_blip2_prompt(question, options)

                    image_path = valid_dataset.image_dir / item["image"]
                    with Image.open(image_path) as image_file:
                        image = image_file.convert("RGB")

                    if options:
                        scoring_targets, mode = build_option_scoring_targets(
                            options=options,
                            gold_answer=gold_answer,
                        )

                        losses = []
                        for target in scoring_targets:
                            loss = _score_blip2_answer(
                                model=model,
                                processor=processor,
                                image=image,
                                prompt=prompt,
                                answer=str(target),
                                device=device,
                                bf16=bf16,
                            )
                            losses.append(loss)

                        best_idx = min(range(len(losses)), key=lambda i: losses[i])

                        pred_answer = convert_best_index_to_prediction(
                            best_idx=best_idx,
                            options=options,
                            gold_answer=gold_answer,
                            mode=mode,
                        )

                        option_losses = {
                            str(scoring_targets[i]): losses[i]
                            for i in range(len(scoring_targets))
                        }

                    else:
                        inputs = processor(
                            images=image,
                            text=prompt,
                            return_tensors="pt",
                        ).to(device)

                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                        )

                        pred_answer = (
                            decode_blip2_output(processor, outputs[0], prompt)
                            or "--"
                        )
                        option_losses = {}
                        mode = "generate"

                    predictions.append(
                        build_result_record(item, idx, pred_answer)
                    )

                    if len(debug_examples) < 10:
                        debug_examples.append(
                            {
                                "id": item.get("id", idx),
                                "question": question,
                                "options": options,
                                "gold": gold_answer,
                                "pred": pred_answer,
                                "mode": mode,
                                "option_losses": option_losses,
                                "gold_norm": normalize_answer_text(gold_answer),
                                "pred_norm": normalize_answer_text(pred_answer),
                            }
                        )

    finally:
        processor.tokenizer.padding_side = original_padding_side

    logger.info("========== DEV ACC DEBUG EXAMPLES ==========")
    for example in debug_examples:
        logger.info(
            "id=%s | mode=%s | gold=%r | pred=%r | gold_norm=%r | pred_norm=%r | losses=%s",
            example["id"],
            example["mode"],
            example["gold"],
            example["pred"],
            example["gold_norm"],
            example["pred_norm"],
            example["option_losses"],
        )
        logger.info("options=%s", example["options"])

    metrics = calculate_spatial_metrics(predictions)

    logger.info("Dev metrics: %s", metrics)

    return metrics["accuracy"]


# ============================================================
# Training
# ============================================================

def run_train(args, config: ExperimentConfig):
    set_seed(config.training.seed)

    out_checkpoint = (
        Path(args.out_checkpoint)
        if args.out_checkpoint
        else Path(config.training.output_dir)
    )
    out_results = (
        Path(args.out_results)
        if args.out_results
        else out_checkpoint
    )

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    logger.info("Building BLIP-2 model and processor...")

    processor = Blip2Processor.from_pretrained(
        config.model.model_name_or_path
    )

    kwargs = {
        "device_map": config.model.device_map,
    }

    if config.model.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif config.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name_or_path,
        **kwargs,
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

    logger.info("Loading BLIP-2 train dataset from %s", train_path)

    train_dataset = ImageCaptioningDataset(
        data_path=str(train_path),
        image_dir=image_dir,
        processor=processor,
        max_samples=config.dataset.max_samples,
    )

    logger.info("Loading BLIP-2 valid dataset from %s", val_path)

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
        shuffle=True,
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
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.9,
        last_epoch=-1,
    )

    # Note:
    # GradScaler is mainly useful with fp16.
    # If bf16=True, scaler can be disabled safely.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(torch.cuda.is_available() and not config.training.bf16),
    )

    min_eval_loss = float("inf")
    best_dev_accuracy = float("-inf")
    early_stopping_hook = 0

    losses_history = []
    dev_loss_history = []
    dev_accuracy_history = []
    log_history = []

    global_step = 0
    grad_accum_steps = max(int(config.training.cal_num), 1)

    logger.info("Starting BLIP-2 training loop...")

    for epoch in range(config.training.num_epochs):
        model.train()

        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.training.num_epochs} [Train]",
        )

        for idx, batch in enumerate(pbar):
            batch_copy = {
                key: value.clone()
                for key, value in batch.items()
            }

            loss = compute_blip2_loss(
                model=model,
                batch=batch_copy,
                device=device,
                bf16=config.training.bf16,
            )

            raw_loss_value = loss.item()
            epoch_loss += raw_loss_value

            loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            should_step = (
                (idx + 1) % grad_accum_steps == 0
                or idx == len(train_dataloader) - 1
            )

            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": raw_loss_value})

            losses_history.append(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": raw_loss_value,
                }
            )

            global_step += 1

        train_loss = epoch_loss / max(len(train_dataloader), 1)

        eval_loss = compute_eval_loss(
            model=model,
            dataloader=valid_dataloader,
            device=device,
            bf16=config.training.bf16,
        )

        dev_loss_history.append(
            {
                "epoch": epoch + 1,
                "eval_loss": eval_loss,
            }
        )

        min_eval_loss = min(min_eval_loss, eval_loss)

        dev_accuracy = compute_dev_accuracy(
            model=model,
            valid_dataset=valid_dataset,
            processor=processor,
            device=device,
            bf16=config.training.bf16,
            batch_size=batch_size,
        )

        dev_accuracy_history.append(
            {
                "epoch": epoch + 1,
                "dev_accuracy": dev_accuracy,
            }
        )

        log_item = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "dev_accuracy": dev_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }

        log_history.append(log_item)

        logger.info(
            "Epoch %s | Train Loss: %.4f | Eval Loss: %.4f | Dev Acc: %.4f | LR: %s",
            epoch + 1,
            train_loss,
            eval_loss,
            dev_accuracy,
            optimizer.param_groups[0]["lr"],
        )

        scheduler.step()

        # Select best model based on dev accuracy, not loss.
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            early_stopping_hook = 0

            save_path = out_checkpoint / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

            logger.info(
                "Saved best model with dev accuracy %.4f to %s",
                dev_accuracy,
                save_path,
            )

        else:
            early_stopping_hook += 1

            if early_stopping_hook > config.training.patience:
                logger.info(
                    "Early stopping triggered after %s epochs.",
                    epoch + 1,
                )
                break

        with open(out_results / "losses.json", "w", encoding="utf-8") as file:
            json.dump(losses_history, file, indent=4, ensure_ascii=False)

        with open(out_results / "dev_loss.json", "w", encoding="utf-8") as file:
            json.dump(dev_loss_history, file, indent=4, ensure_ascii=False)

        with open(out_results / "dev_accuracy.json", "w", encoding="utf-8") as file:
            json.dump(dev_accuracy_history, file, indent=4, ensure_ascii=False)

        with open(out_results / "log.json", "w", encoding="utf-8") as file:
            json.dump(log_history, file, indent=4, ensure_ascii=False)

    with open(out_results / "best_dev_metric.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_dev_accuracy": best_dev_accuracy,
                "best_dev_loss": min_eval_loss,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "BLIP-2 training complete. Loading best model for evaluation on test set..."
    )

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