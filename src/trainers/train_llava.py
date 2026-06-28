"""
Training wrapper for LLaVA (liuhaotian/llava-v1.5-7b).

Generates a deepspeed training script matching the reference SpatialMQA
llava_lora_train.sh, then runs it via subprocess.
"""

import shlex
import subprocess
import sys
from pathlib import Path

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import get_sample_id, resolve_split_paths
from ..utils.io import load_json, load_jsonl, save_json
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# Path to the LLaVA repository (cloned via scripts/setup_llava.sh)
LLAVA_REPO = Path("/workspace/LLaVA")


def convert_to_llava_format(data_path: str, output_path: str) -> None:
    path = Path(data_path)
    if path.suffix == ".jsonl":
        data = load_jsonl(path)
    else:
        data = load_json(path)

    llava_data = []
    for index, item in enumerate(data):
        llava_item = {
            "id": get_sample_id(item, index),
            "image": item["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": str(item["answer"])},
            ],
        }
        llava_data.append(llava_item)

    save_json(llava_data, output_path)


def run_train(args, config: ExperimentConfig):
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    data_path = args.jsonl_dir or config.dataset.data_path
    image_dir = args.image_dir or config.dataset.image_dir
    train_path, _ = resolve_split_paths(data_path)
    formatted_data_path = out_results / "llava_train_data.json"

    logger.info(f"Formatting dataset to LLaVA conversational format: {formatted_data_path}")
    convert_to_llava_format(str(train_path), str(formatted_data_path))

    saved_model_dir = out_checkpoint / "saved_model"

    script_path = out_checkpoint / "train_llava.sh"
    script = f"""#!/bin/bash
set -e
cd {shlex.quote(str(LLAVA_REPO))}

deepspeed --include localhost:0 llava/train/train.py \\
    --lora_enable True --lora_r {config.model.lora_r} --lora_alpha {config.model.lora_alpha} \\
    --mm_projector_lr {getattr(config.model, 'mm_projector_lr', 2e-5)} \\
    --deepspeed ./scripts/zero3.json \\
    --model_name_or_path {config.model.model_name_or_path} \\
    --version v1 \\
    --data_path {shlex.quote(str(formatted_data_path.resolve()))} \\
    --image_folder {shlex.quote(str(Path(image_dir).resolve()))} \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --bf16 {str(config.training.bf16).lower()} \\
    --output_dir {shlex.quote(str(saved_model_dir.resolve()))} \\
    --num_train_epochs {config.training.num_epochs} \\
    --per_device_train_batch_size {config.training.batch_size} \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps {getattr(config.training, 'gradient_accumulation_steps', 2)} \\
    --evaluation_strategy "no" \\
    --save_steps 100 \\
    --save_total_limit 3 \\
    --learning_rate {config.training.learning_rate} \\
    --weight_decay {getattr(config.training, 'weight_decay', 0.0)} \\
    --warmup_ratio {getattr(config.training, 'warmup_ratio', 0.02)} \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 2 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 0 \\
    --lazy_preprocess True
"""
    script_path.write_text(script, encoding="utf-8")
    logger.info("Generated LLaVA training script: %s", script_path)

    if not LLAVA_REPO.exists():
        logger.error(
            "LLaVA repo not found at %s. Run: bash scripts/setup_llava.sh %s",
            LLAVA_REPO, LLAVA_REPO,
        )
        sys.exit(1)

    logger.info("Starting LLaVA LoRA training via deepspeed...")
    result = subprocess.run(
        ["bash", str(script_path.resolve())],
        cwd=str(LLAVA_REPO),
        check=False,
    )
    if result.returncode != 0:
        logger.error("LLaVA training failed with return code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("LLaVA training completed successfully!")
