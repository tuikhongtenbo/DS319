"""
Configuration classes using dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import yaml

@dataclass
class ModelConfig:
    model_name_or_path: str
    model_type: str  # 'blip2-opt-2.7b', 'llava-v1.5-7b', 'spacellava', 'gpt-4o', 'qwen-3.6'
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # vLLM inference (faster, uses PagedAttention / TensorParallel)
    use_vllm: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    # LoRA config
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

@dataclass
class DatasetConfig:
    data_path: str
    image_dir: str
    max_samples: Optional[int] = None

@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int = 10  # matches reference
    batch_size: int = 8
    learning_rate: float = 2e-4  # matches reference
    gradient_accumulation_steps: int = 2  # matches reference (effective batch = 8*2=16)
    patience: int = 10
    seed: int = 42
    cal_num: int = 2  # kept for backward compat
    bf16: bool = True
    warmup_ratio: float = 0.02  # matches reference
    weight_decay: float = 0.0  # matches reference

@dataclass
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    training: Optional[TrainingConfig] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        p = Path(yaml_path)
        if not p.exists():
            # Try src/configs/ prefix (common in this repo)
            alt = Path("src/configs") / p.name
            if alt.exists():
                yaml_path = str(alt)
            else:
                raise FileNotFoundError(
                    f"Config not found: {yaml_path}\n"
                    f"Tried: {alt}"
                )
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        model_cfg = ModelConfig(**data.get('model', {}))
        dataset_cfg = DatasetConfig(**data.get('dataset', {}))
        
        training_data = data.get('training')
        training_cfg = TrainingConfig(**training_data) if training_data else None
        
        return cls(model=model_cfg, dataset=dataset_cfg, training=training_cfg)
