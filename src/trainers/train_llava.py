"""
Training wrapper for LLaVA (liuhaotian/llava-v1.5-7b).

Trains epoch-by-epoch via deepspeed, evaluates on dev set after each epoch,
saves the best model (overwrite), and finally evaluates on test set.
"""

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import get_sample_id, resolve_split_paths, resolve_test_path
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


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint directory inside output_dir."""
    if not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0]
    if (output_dir / "adapter_config.json").exists():
        return output_dir
    return None


def _run_eval(checkpoint_path: Path, eval_data_path: Path, image_dir: str,
              config: ExperimentConfig, results_dir: Path) -> float:
    """Run inference on a dataset split and return accuracy."""
    from ..inference.inference_llava import run_infer as llava_infer

    results_dir.mkdir(parents=True, exist_ok=True)

    class EvalArgs:
        out_checkpoint = str(checkpoint_path)
        out_results = str(results_dir)
        jsonl_dir = str(eval_data_path)  # direct path to dev.jsonl or test.jsonl
        image_dir_val = image_dir

    eval_args = EvalArgs()
    eval_args.image_dir = image_dir
    return llava_infer(eval_args, config)


def _generate_ds_config(out_checkpoint: Path) -> Path:
    """Generate a DeepSpeed ZeRO-2 config (better for LoRA + single GPU)."""
    ds_config_path = out_checkpoint / "ds_zero2.json"
    ds_config = {
        "fp16": {"enabled": "auto"},
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 50000000,
            "allgather_bucket_size": 50000000,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    save_json(ds_config, str(ds_config_path))
    return ds_config_path


def _generate_train_script(
    script_path: Path,
    ds_config_path: Path,
    formatted_data_path: Path,
    image_dir: str,
    saved_model_dir: Path,
    config: ExperimentConfig,
    cumulative_epochs: int,
    resume_checkpoint: Path | None = None,
) -> None:
    """Generate a deepspeed training bash script for a given epoch range."""
    resume_line = ""
    if resume_checkpoint is not None:
        resume_line = f"    --resume_from_checkpoint {shlex.quote(str(resume_checkpoint))} \\\\"

    script = f"""#!/bin/bash
set -e
cd {shlex.quote(str(LLAVA_REPO))}

deepspeed --include localhost:0 llava/train/train.py \\
    --lora_enable True --lora_r {config.model.lora_r} --lora_alpha {config.model.lora_alpha} \\
    --deepspeed {shlex.quote(str(ds_config_path.resolve()))} \\
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
    --num_train_epochs {cumulative_epochs} \\
    --per_device_train_batch_size {config.training.batch_size} \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps {config.training.cal_num} \\
    --evaluation_strategy "no" \\
    --save_strategy "epoch" \\
    --save_total_limit 1 \\
    --learning_rate {config.training.learning_rate} \\
    --weight_decay 0. \\
    --warmup_ratio 0.02 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 2 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 0 \\
    --lazy_preprocess True \\
{resume_line}
"""
    script_path.write_text(script, encoding="utf-8")


def run_train(args, config: ExperimentConfig):
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    data_path = args.jsonl_dir or config.dataset.data_path
    image_dir = args.image_dir or config.dataset.image_dir
    train_path, val_path = resolve_split_paths(data_path)
    test_path = resolve_test_path(data_path)
    formatted_data_path = out_results / "llava_train_data.json"

    logger.info(f"Formatting dataset to LLaVA conversational format: {formatted_data_path}")
    convert_to_llava_format(str(train_path), str(formatted_data_path))

    # Check if dev set exists (val_path != train_path means separate dev.jsonl)
    has_dev = val_path.exists() and val_path != train_path
    if has_dev:
        logger.info(f"Dev set found: {val_path}")
    else:
        logger.warning("No separate dev set found. Will skip per-epoch dev evaluation.")

    # ── Setup ───────────────────────────────────────────────────────────
    saved_model_dir = out_checkpoint / "saved_model"
    best_model_dir = out_checkpoint / "best_model"
    ds_config_path = _generate_ds_config(out_checkpoint)
    script_path = out_checkpoint / "train_llava.sh"
    num_epochs = config.training.num_epochs
    patience = config.training.patience

    if not LLAVA_REPO.exists():
        logger.error(
            "LLaVA repo not found at %s. Run: bash scripts/setup_llava.sh %s",
            LLAVA_REPO, LLAVA_REPO,
        )
        sys.exit(1)

    # ── Epoch-by-epoch training loop ────────────────────────────────────
    best_accuracy = -1.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        logger.info("=" * 60)
        logger.info(f"  EPOCH {epoch}/{num_epochs}")
        logger.info("=" * 60)

        # Find resume checkpoint (for epoch > 1)
        resume_ckpt = _find_latest_checkpoint(saved_model_dir) if epoch > 1 else None

        # Generate training script for this epoch
        _generate_train_script(
            script_path=script_path,
            ds_config_path=ds_config_path,
            formatted_data_path=formatted_data_path,
            image_dir=image_dir,
            saved_model_dir=saved_model_dir,
            config=config,
            cumulative_epochs=epoch,
            resume_checkpoint=resume_ckpt,
        )

        # Run training for 1 epoch
        logger.info(f"Training epoch {epoch} via deepspeed...")
        result = subprocess.run(
            ["bash", str(script_path.resolve())],
            cwd=str(LLAVA_REPO),
            check=False,
        )
        if result.returncode != 0:
            logger.error("Training failed at epoch %d (return code %d)", epoch, result.returncode)
            break

        logger.info(f"Epoch {epoch} training completed.")

        # ── Dev evaluation ──────────────────────────────────────────────
        checkpoint = _find_latest_checkpoint(saved_model_dir)
        if checkpoint is None:
            logger.warning("No checkpoint found after epoch %d", epoch)
            continue

        if has_dev:
            logger.info(f"Evaluating epoch {epoch} on dev set: {val_path}")
            dev_results_dir = out_results / f"dev_epoch_{epoch}"
            accuracy = _run_eval(checkpoint, val_path, image_dir, config, dev_results_dir)
            logger.info(f"Epoch {epoch} dev accuracy: {accuracy:.4f} (best so far: {best_accuracy:.4f})")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                # Save best model (overwrite)
                if best_model_dir.exists():
                    shutil.rmtree(best_model_dir)
                shutil.copytree(str(checkpoint), str(best_model_dir))
                logger.info(f"★ New best model saved at epoch {epoch}! Accuracy: {accuracy:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        else:
            # No dev set: always keep latest as best
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            shutil.copytree(str(checkpoint), str(best_model_dir))
            logger.info(f"Model checkpoint saved at epoch {epoch} (no dev eval).")

    # ── Final evaluation on test set ────────────────────────────────────
    final_model = best_model_dir if best_model_dir.exists() else _find_latest_checkpoint(saved_model_dir)
    if final_model is None:
        logger.error("No model found for final evaluation.")
        return

    logger.info("=" * 60)
    logger.info("  FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)
    logger.info(f"Using model: {final_model}")
    logger.info(f"Test data: {test_path}")

    _run_eval(final_model, test_path, image_dir, config, out_results)
