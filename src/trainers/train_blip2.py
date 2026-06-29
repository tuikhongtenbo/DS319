import os
import re
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================
# Config
# ============================================================

model_dir = "/projects/Models/"
model_name_or_path = f"{model_dir}blip2-opt-2.7b"

image_dir = "/projects/SpatialMQA/COCO2017/test2017/"

train_file = "/projects/SpatialMQA/datasets/blip2_train/train_3780.jsonl"
dev_file = "/projects/SpatialMQA/datasets/blip2_train/dev_536.jsonl"

output_dir = Path("/projects/SpatialMQA/finetune_models/models_arg/blip2_lora_best_acc")
output_dir.mkdir(parents=True, exist_ok=True)

cuda_id = 7
device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

batch_size = 8
learning_rate = 4e-5
num_epochs = 30
patience = 5
grad_accum_steps = 2
bf16 = True

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(42)


# ============================================================
# Load processor and dataset
# ============================================================

processor = Blip2Processor.from_pretrained(model_name_or_path)

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

train_ds = load_dataset(
    "json",
    data_files=train_file,
    split="train[:100%]",
)

eval_ds = load_dataset(
    "json",
    data_files=dev_file,
    split="train[:100%]",
)

print(f"Training sets: {len(train_ds)} - Validating set: {len(eval_ds)}")


# ============================================================
# Utility functions
# ============================================================

def normalize_answer(text):
    """
    Normalize text để so sánh prediction với gold answer.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.replace("</s>", "")
    text = text.replace("<pad>", "")
    text = text.replace("<s>", "")

    # Nếu model generate kiểu "Answer: left"
    text = re.sub(r"^answer\s*[:\-]\s*", "", text)

    # Chỉ lấy dòng đầu tiên nếu model generate nhiều dòng
    text = text.split("\n")[0].strip()

    # Xóa dấu câu đơn giản ở đầu/cuối
    text = text.strip(" .,:;!?")

    # Gộp khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def decode_blip2_output(processor, output_ids, prompt):
    """
    Decode output của BLIP-2.

    Một số model generate có thể trả lại cả prompt + answer,
    nên hàm này có bước strip prompt nếu cần.
    """
    text = processor.tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    prompt = str(prompt).strip()

    if text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()

    return text


# ============================================================
# Dataset
# ============================================================

class ImageCaptioningDataset(Dataset):
    """
    Dataset BLIP-2 giống code tác giả:

    Input:
        image + question

    Label:
        answer + eos_token

    Không dùng options, không dùng build_blip2_prompt.
    """

    def __init__(self, dataset, image_dir, processor):
        self.dataset = dataset
        self.image_dir = Path(image_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        question = item["question"]
        answer = str(item["answer"])
        image_name = item["image"]

        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")

        return {
            "idx": idx,
            "image": image,
            "question": question,
            "answer": answer,
            "raw_item": item,
        }


def blip2_collate_fn(batch):
    """
    Collate function để batch được các question/answer có độ dài khác nhau.

    Giống tác giả ở logic:
        processor(images=image, text=question)
        labels = answer + eos

    Nhưng viết chắc hơn để tránh lỗi stack tensor do độ dài khác nhau.
    """
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    encoding = processor(
        images=images,
        text=questions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    eos_token_id = processor.tokenizer.eos_token_id

    if eos_token_id is None:
        # BLIP-2 OPT thường dùng 50118
        eos_token_id = 50118

    label_ids = []

    for answer in answers:
        ids = processor.tokenizer(
            str(answer),
            add_special_tokens=False,
        ).input_ids

        ids = ids + [eos_token_id]
        label_ids.append(ids)

    max_label_len = max(len(ids) for ids in label_ids)

    padded_labels = []

    for ids in label_ids:
        pad_len = max_label_len - len(ids)

        # -100 để HuggingFace loss bỏ qua token padding
        padded = ids + [-100] * pad_len
        padded_labels.append(padded)

    labels = torch.tensor(padded_labels, dtype=torch.long)

    encoding["labels"] = labels

    return encoding


# ============================================================
# Load model
# ============================================================

print("Loading BLIP-2 model...")

if torch.cuda.is_available():
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map={"": cuda_id},
        load_in_8bit=True,
    )
else:
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path,
    )
    model = model.to(device)

# Nên có khi fine-tune 8-bit + LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

if not torch.cuda.is_available():
    model = model.to(device)


# ============================================================
# DataLoader
# ============================================================

train_dataset = ImageCaptioningDataset(
    dataset=train_ds,
    image_dir=image_dir,
    processor=processor,
)

valid_dataset = ImageCaptioningDataset(
    dataset=eval_ds,
    image_dir=image_dir,
    processor=processor,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=blip2_collate_fn,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=blip2_collate_fn,
)


# ============================================================
# Loss / Eval Loss
# ============================================================

def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def compute_eval_loss(model, dataloader, device):
    model.eval()

    eval_loss = 0.0
    dtype = torch.bfloat16 if bf16 else torch.float16
    autocast_enabled = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating batch"):
            batch = move_batch_to_device(batch, device)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss

            eval_loss += loss.item()

    return eval_loss / max(len(dataloader), 1)


# ============================================================
# Dev Accuracy
# ============================================================

def compute_dev_accuracy(
    model,
    valid_dataset,
    processor,
    device,
    max_new_tokens=20,
):
    """
    Dev accuracy giống BLIP-1 và giống style tác giả:

    Input:
        image + question

    Predict:
        model.generate()

    Compare:
        normalize(pred) == normalize(gold)
    """
    model.eval()

    correct = 0
    total = 0
    debug_examples = []

    dtype = torch.bfloat16 if bf16 else torch.float16
    autocast_enabled = device.type == "cuda"

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = (
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
            or 1
        )

    with torch.no_grad():
        for idx in tqdm(range(len(valid_dataset)), desc="Dev Accuracy"):
            item = valid_dataset.dataset[idx]

            question = item["question"]
            gold_answer = str(item["answer"])

            image_name = item["image"]
            image_path = valid_dataset.image_dir / image_name

            image = Image.open(image_path).convert("RGB")

            # Quan trọng:
            # Giống code tác giả: chỉ dùng question, không dùng options.
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt",
            ).to(device)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            decoded = decode_blip2_output(
                processor=processor,
                output_ids=outputs[0],
                prompt=question,
            )

            pred_answer = normalize_answer(decoded)
            gold_norm = normalize_answer(gold_answer)

            is_correct = pred_answer == gold_norm

            correct += int(is_correct)
            total += 1

            if len(debug_examples) < 10:
                debug_examples.append(
                    {
                        "idx": idx,
                        "question": question,
                        "gold": gold_answer,
                        "decoded": decoded,
                        "pred_norm": pred_answer,
                        "gold_norm": gold_norm,
                        "correct": is_correct,
                    }
                )

    accuracy = correct / max(total, 1)

    print("\n========== DEV ACC DEBUG EXAMPLES ==========")
    for ex in debug_examples:
        print(
            f"idx={ex['idx']} | "
            f"gold={ex['gold']!r} | "
            f"decoded={ex['decoded']!r} | "
            f"pred_norm={ex['pred_norm']!r} | "
            f"gold_norm={ex['gold_norm']!r} | "
            f"correct={ex['correct']}"
        )
        print(f"question={ex['question']}")
        print("-" * 80)

    print(f"Dev Accuracy: {accuracy:.4f}")

    return accuracy


# ============================================================
# Optimizer / Scheduler
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.9,
    last_epoch=-1,
)

# bf16 thường không cần GradScaler
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda" and not bf16),
)


# ============================================================
# Training loop
# ============================================================

best_dev_accuracy = float("-inf")
min_eval_loss = float("inf")
early_stopping_hook = 0

tracking_information = []
losses_history = []
dev_loss_history = []
dev_accuracy_history = []
log_history = []

global_step = 0

print("Starting BLIP-2 LoRA finetuning...")

for epoch in range(num_epochs):
    model.train()

    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
    )

    for idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)

        dtype = torch.bfloat16 if bf16 else torch.float16
        autocast_enabled = device.type == "cuda"

        with torch.amp.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

        raw_loss = loss.item()
        epoch_loss += raw_loss

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

        pbar.set_postfix({"loss": raw_loss})

        losses_history.append(
            {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": raw_loss,
            }
        )

        global_step += 1

    train_loss = epoch_loss / max(len(train_dataloader), 1)

    eval_loss = compute_eval_loss(
        model=model,
        dataloader=valid_dataloader,
        device=device,
    )

    min_eval_loss = min(min_eval_loss, eval_loss)

    dev_accuracy = compute_dev_accuracy(
        model=model,
        valid_dataset=valid_dataset,
        processor=processor,
        device=device,
        max_new_tokens=20,
    )

    scheduler.step()

    tracking_information.append(
        (
            train_loss,
            eval_loss,
            dev_accuracy,
            optimizer.param_groups[0]["lr"],
        )
    )

    dev_loss_history.append(
        {
            "epoch": epoch + 1,
            "eval_loss": eval_loss,
        }
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
        "best_dev_accuracy": best_dev_accuracy,
        "lr": optimizer.param_groups[0]["lr"],
    }

    log_history.append(log_item)

    print(
        f"Epoch: {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Eval Loss: {eval_loss:.4f} | "
        f"Dev Acc: {dev_accuracy:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']}"
    )

    # ========================================================
    # Save best model theo dev accuracy
    # ========================================================

    if dev_accuracy > best_dev_accuracy:
        best_dev_accuracy = dev_accuracy
        early_stopping_hook = 0

        best_model_path = output_dir / "best_model"
        best_model_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(best_model_path)
        processor.save_pretrained(best_model_path)

        print(
            f"Saved best model to {best_model_path} "
            f"with dev accuracy = {best_dev_accuracy:.4f}"
        )

    else:
        early_stopping_hook += 1

        if early_stopping_hook > patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # ========================================================
    # Save logs mỗi epoch
    # ========================================================

    with open(output_dir / "losses.json", "w", encoding="utf-8") as f:
        json.dump(losses_history, f, indent=4, ensure_ascii=False)

    with open(output_dir / "dev_loss.json", "w", encoding="utf-8") as f:
        json.dump(dev_loss_history, f, indent=4, ensure_ascii=False)

    with open(output_dir / "dev_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(dev_accuracy_history, f, indent=4, ensure_ascii=False)

    with open(output_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=4, ensure_ascii=False)

    with open(output_dir / "tracking_information.pkl", "wb") as f:
        pickle.dump(tracking_information, f)


# ============================================================
# Save final metric
# ============================================================

with open(output_dir / "best_dev_metric.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_dev_accuracy": best_dev_accuracy,
            "best_dev_loss": min_eval_loss,
        },
        f,
        indent=4,
        ensure_ascii=False,
    )

print("The finetuning process has done!")
print(f"Best Dev Accuracy: {best_dev_accuracy:.4f}")
print(f"Best Eval Loss: {min_eval_loss:.4f}")