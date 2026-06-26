"""
Training wrapper for SpaceLLaVA.
Generates a bash script compatible with SpatialMQA / SpaceLLaVA training format.
"""

import shlex
from pathlib import Path

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import get_sample_id, resolve_split_paths
from ..utils.io import load_json, load_jsonl, save_json
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def convert_to_spacellava_format(data_path: str, output_path: str) -> None:
    path = Path(data_path)
    if path.suffix == ".jsonl":
        data = load_jsonl(path)
    else:
        data = load_json(path)

    spacellava_data = []
    for index, item in enumerate(data):
        spacellava_item = {
            "id": get_sample_id(item, index),
            "image": item["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": str(item["answer"])},
            ],
        }
        spacellava_data.append(spacellava_item)

    save_json(spacellava_data, output_path)


def run_train(args, config: ExperimentConfig):
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    data_path = args.jsonl_dir or config.dataset.data_path
    image_dir = args.image_dir or config.dataset.image_dir
    train_path, _ = resolve_split_paths(data_path)
    formatted_data_path = out_results / "spacellava_train_data.json"

    logger.info(f"Formatting dataset to SpaceLLaVA conversational format: {formatted_data_path}")
    convert_to_spacellava_format(str(train_path), str(formatted_data_path))

    script_path = out_checkpoint / "train_spacellava.sh"
    script = f"""#!/bin/bash

deepspeed --include localhost:0 spacellava/train/train_mem.py \\
    --lora_enable True \\
    --lora_r {config.model.lora_r} \\
    --lora_alpha {config.model.lora_alpha} \\
    --mm_projector_lr 2e-5 \\
    --deepspeed ./scripts/zero3.json \\
    --model_name_or_path {config.model.model_name_or_path} \\
    --version v1 \\
    --data_path {shlex.quote(str(formatted_data_path))} \\
    --image_folder {shlex.quote(str(image_dir))} \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --bf16 {str(config.training.bf16).lower()} \\
    --output_dir {shlex.quote(str(out_checkpoint / 'saved_model'))} \\
    --num_train_epochs {config.training.num_epochs} \\
    --per_device_train_batch_size {config.training.batch_size} \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps {config.training.cal_num} \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 100 \\
    --save_total_limit 1 \\
    --learning_rate {config.training.learning_rate} \\
    --weight_decay 0. \\
    --warmup_ratio 0.02 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 10 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 0 \\
    --lazy_preprocess True
"""
    script_path.write_text(script, encoding="utf-8")
    logger.info("Generated SpaceLLaVA training script: %s", script_path)
    logger.info("Run this script inside the SpaceLLaVA repository environment.")
