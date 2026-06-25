"""
LLaVA training wrapper.
"""

import logging
from pathlib import Path
from ..datasets.dataset import prepare_llava_dataset

logger = logging.getLogger(__name__)

class LLaVATrainerWrapper:
    """
    Since LLaVA/SpaceLLaVA requires specific deepspeed and trainer wrappers 
    from the original llava repository, this class will format the data 
    and generate the bash script needed to kick off training.
    """
    def __init__(self, data_path: str, image_dir: str, output_dir: str, model_path: str, is_spacellava: bool = False):
        self.data_path = data_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.is_spacellava = is_spacellava

    def prepare_and_generate_script(self):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Convert dataset to LLaVA format
        formatted_data_path = output_dir / "llava_formatted_data.json"
        logger.info(f"Formatting dataset to LLaVA format: {formatted_data_path}")
        prepare_llava_dataset(self.data_path, str(formatted_data_path))
        
        # 2. Generate bash script
        script_path = output_dir / "train_llava.sh"
        
        # Script template based on original bash scripts
        script_content = f"""#!/bin/bash
# Auto-generated LLaVA training script
deepspeed --include localhost:0 llava/train/train_mem.py \\
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \\
    --deepspeed ./scripts/zero3.json \\
    --model_name_or_path {self.model_path} \\
    --version v1 \\
    --data_path {str(formatted_data_path)} \\
    --image_folder {self.image_dir} \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --bf16 True \\
    --output_dir {self.output_dir}/saved_model \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 1 \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 60 \\
    --save_total_limit 1 \\
    --learning_rate 2e-4 \\
    --weight_decay 0. \\
    --warmup_ratio 0.02 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 0 \\
    --lazy_preprocess True 
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        logger.info(f"Generated LLaVA training script: {script_path}")
        logger.info("To run LLaVA training, execute this bash script in an environment where LLaVA is installed.")
