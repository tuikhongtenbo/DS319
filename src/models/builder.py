"""
Model and processor building utilities.
"""

from typing import Tuple, Any
import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForQuestionAnswering,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model

from ..configs.config import ModelConfig


def build_model_and_processor(config: ModelConfig) -> Tuple[Any, Any]:
    """
    Builds the model and processor based on the configuration.
    """
    model_type = config.model_type.lower()
    model_name_or_path = config.model_name_or_path
    
    kwargs = {
        "device_map": config.device_map,
    }
    
    if config.load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif config.load_in_4bit:
        kwargs["load_in_4bit"] = True
    
    # Initialize processor and model based on type
    if "blip-vqa-base" in model_type:
        processor = BlipProcessor.from_pretrained(model_name_or_path)
        model = BlipForQuestionAnswering.from_pretrained(model_name_or_path, **kwargs)
    elif "blip2" in model_type:
        processor = Blip2Processor.from_pretrained(model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    elif "instructblip" in model_type:
        processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    elif "llava" in model_type or "spacellava" in model_type or "mplug" in model_type:
        # LLaVA and mPLUG-Owl models usually can be loaded with Auto classes 
        # or require trust_remote_code=True depending on version
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Wrap model with LoRA if specified
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=config.lora_target_modules
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor
