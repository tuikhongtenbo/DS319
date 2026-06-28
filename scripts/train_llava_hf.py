#!/usr/bin/env python3
"""
Train LLaVA LoRA - wrapper that calls LLaVA's train.py directly.
"""

import sys
from pathlib import Path

# Path to LLaVA repo
LLAVA_REPO = Path("/workspace/LLaVA")

# Add LLaVA to path
sys.path.insert(0, str(LLAVA_REPO))

# Import the train function from LLaVA
# LLaVA's train.py has a main() or train() function at the bottom
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_enable", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
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
    parser.add_argument("--group_by_modality_length", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
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
    parser1 = parser
    
    args = parser1.parse_args()
    
    # Build command to run LLaVA train.py directly
    import subprocess
    
    cmd = [
        sys.executable,
        str(LLAVA_REPO / "llava" / "train" / "train.py"),
        "--lora_enable", str(args.lora_enable),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--model_name_or_path", args.model_name_or_path,
        "--version", args.version,
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--vision_tower", args.vision_tower,
        "--mm_projector_type", args.mm_projector_type,
        "--mm_vision_select_layer", str(args.mm_vision_select_layer),
        "--mm_use_im_start_end", str(args.mm_use_im_start_end),
        "--mm_use_im_patch_token", str(args.mm_use_im_patch_token),
        "--image_aspect_ratio", args.image_aspect_ratio,
        "--group_by_modality_length", str(args.group_by_modality_length),
        "--bf16", str(args.bf16),
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--evaluation_strategy", args.evaluation_strategy,
        "--save_strategy", args.save_strategy,
        "--save_steps", str(args.save_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--logging_steps", str(args.logging_steps),
        "--model_max_length", str(args.model_max_length),
        "--gradient_checkpointing", str(args.gradient_checkpointing),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--lazy_preprocess", str(args.lazy_preprocess),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
