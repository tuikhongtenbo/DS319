#!/usr/bin/env python3
"""
Train LLaVA LoRA without flash attention.

Usage:
    python scripts/train_llava_hf.py --config train_llava.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add LLaVA to path
LLAVA_REPO = Path("/workspace/LLaVA")
sys.path.insert(0, str(LLAVA_REPO))

from llava.train.trainer import train
from llava.utils.config import training_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_enable", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--mm_use_im_start_end", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--mm_use_im_patch_token", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--group_by_modulate_length", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True, help="Use bfloat16")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--lazy_preprocess", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # Build training args compatible with original train.py
    training_args.output_dir = args.output_dir
    training_args.lora_enable = args.lora_enable
    training_args.lora_r = args.lora_r
    training_args.lora_alpha = args.lora_alpha
    training_args.lora_dropout = args.lora_dropout
    training_args.lora_target_modules = args.lora_target_modules.split(",")
    training_args.model_name_or_path = args.model_name_or_path
    training_args.version = args.version
    training_args.data_path = args.data_path
    training_args.image_folder = args.image_folder
    training_args.vision_tower = args.vision_tower
    training_args.mm_projector_type = args.mm_projector_type
    training_args.mm_vision_select_layer = args.mm_vision_select_layer
    training_args.mm_use_im_start_end = args.mm_use_im_start_end
    training_args.mm_use_im_patch_token = args.mm_use_im_patch_token
    training_args.image_aspect_ratio = args.image_aspect_ratio
    training_args.group_by_modality_length = args.group_by_modulate_length
    training_args.bf16 = args.bf16
    training_args.num_train_epochs = args.num_train_epochs
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.evaluation_strategy = args.evaluation_strategy
    training_args.save_strategy = args.save_strategy
    training_args.save_steps = args.save_steps
    training_args.save_total_limit = args.save_total_limit
    training_args.learning_rate = args.learning_rate
    training_args.weight_decay = args.weight_decay
    training_args.warmup_ratio = args.warmup_ratio
    training_args.lr_scheduler_type = args.lr_scheduler_type
    training_args.logging_steps = args.logging_steps
    training_args.model_max_length = args.model_max_length
    training_args.gradient_checkpointing = args.gradient_checkpointing
    training_args.dataloader_num_workers = args.dataloader_num_workers
    training_args.local_rank = args.local_rank

    # DON'T use flash attention - causes compatibility issues
    # Just use default attention
    train(attn_implementation="eager")


if __name__ == "__main__":
    main()
