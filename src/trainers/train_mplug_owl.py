"""
Training wrapper for mPLUG-Owl.
Generates the bash script required for mPLUG-Owl finetuning.
"""

from pathlib import Path

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import resolve_split_paths
from ..utils.io import load_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def run_train(args, config: ExperimentConfig):
    out_checkpoint = Path(args.out_checkpoint) if args.out_checkpoint else Path(config.training.output_dir)
    out_results = Path(args.out_results) if args.out_results else out_checkpoint

    out_checkpoint.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    data_path = args.jsonl_dir or config.dataset.data_path
    train_path, _ = resolve_split_paths(data_path)
    train_data = load_jsonl(train_path)
    samples = len(train_data)

    epochs = config.training.num_epochs
    micro_batch = config.training.batch_size
    grad_accum = config.training.cal_num
    global_batch = micro_batch * grad_accum
    train_iters = int(samples * epochs / global_batch) if global_batch > 0 else 4700

    script_path = out_checkpoint / "train_mplug_owl.sh"
    script_content = f"""#!/bin/bash
# Auto-generated mPLUG-Owl training script
# Run this inside the mPLUG-Owl repository root

MASTER_ADDR=127.0.0.1
MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
WORLD_SIZE=1
RANK=0

DISTRIBUTED_ARGS="--nproc_per_node 1 \\
                  --nnodes ${{WORLD_SIZE}} \\
                  --node_rank ${{RANK}} \\
                  --master_addr ${{MASTER_ADDR}} \\
                  --master_port ${{MASTER_PORT}}"

SAVE_PATH="{out_checkpoint}/saved_model"
mkdir -p ${{SAVE_PATH}}

options=" \\
    --pretrained-ckpt {config.model.model_name_or_path} \\
    --seq-length 2048 \\
    --micro-batch-size {micro_batch} \\
    --num-training-steps {train_iters} \\
    --train-epochs {epochs} \\
    --num-warmup-steps 50 \\
    --gradient-accumulation-steps {grad_accum} \\
    --lr {config.training.learning_rate} \\
    --min-lr 1e-6 \\
    --eval-iters 50 \\
    --save-interval 500 \\
    --save-path ${{SAVE_PATH}} \\
    --clip-grad 1.0 \\
    --weight-decay 0.0001 \\
    --adam-beta1 0.9 \\
    --adam-beta2 0.999 \\
    --num-workers 8 \\
    --use-lora \\
    --gradient-checkpointing \\
    --bf16 {str(config.training.bf16).lower()}"

multimodal_options=" \\
    --mm-config configs/v0.yaml"

python -m torch.distributed.launch ${{DISTRIBUTED_ARGS}} ./pipeline/train.py \\
    --data_path {train_path} \\
    ${{options}} \\
    ${{multimodal_options}} 2>&1 | tee {out_results}/mplug_owl_train.log
"""
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_content)

    logger.info(f"Generated mPLUG-Owl training script: {script_path}")
    logger.info("Execute this script to start mPLUG-Owl training in your environment.")
