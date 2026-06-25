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
    num_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 4e-5
    patience: int = 5
    seed: int = 42
    cal_num: int = 2
    bf16: bool = True

@dataclass
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    training: Optional[TrainingConfig] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        model_cfg = ModelConfig(**data.get('model', {}))
        dataset_cfg = DatasetConfig(**data.get('dataset', {}))
        
        training_data = data.get('training')
        training_cfg = TrainingConfig(**training_data) if training_data else None
        
        return cls(model=model_cfg, dataset=dataset_cfg, training=training_cfg)
