"""
Training wrapper for LLaVA (liuhaotian/llava-v1.5-7b).

Uses custom HuggingFace Trainer script without flash attention for stability.
"""

import subprocess
import sys
from pathlib import Path

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import get_sample_id, resolve_split_paths
from ..utils.io import load_json, load_jsonl, save_json
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

LLAVA_REPO = Path("/workspace/LLaVA")
SCRIPT_DIR = Path("/workspace/DS319/scripts")


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

    logger.info("Starting LLaVA LoRA training with eager attention (no flash attention)...")

    cmd = [
        sys.executable,  # Use current Python (venv)
        str(SCRIPT_DIR / "train_llava_hf.py"),
        "--lora_enable", "True",
        "--lora_r", str(config.model.lora_r),
        "--lora_alpha", str(config.model.lora_alpha),
        "--model_name_or_path", config.model.model_name_or_path,
        "--data_path", str(formatted_data_path.resolve()),
        "--image_folder", str(Path(image_dir).resolve()),
        "--bf16", str(config.training.bf16).lower(),
        "--output_dir", str(saved_model_dir.resolve()),
        "--num_train_epochs", str(config.training.num_epochs),
        "--per_device_train_batch_size", str(config.training.batch_size),
        "--learning_rate", str(config.training.learning_rate),
        "--weight_decay", str(getattr(config.training, 'weight_decay', 0.0)),
        "--warmup_ratio", str(getattr(config.training, 'warmup_ratio', 0.02)),
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
    ]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("LLaVA training failed with return code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("LLaVA training completed successfully!")
