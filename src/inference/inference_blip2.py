"""
Inference script specialized for BLIP-2 (Salesforce/blip2-opt-2.7b).
"""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import (
    build_blip2_prompt,
    build_result_record,
    decode_blip2_output,
    resolve_test_path,
)
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class Blip2Predictor:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: list) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = build_blip2_prompt(question, options)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        return decode_blip2_output(self.processor, outputs[0], prompt)


def run_infer(args, config: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building BLIP-2 model and processor for inference...")
    processor = Blip2Processor.from_pretrained(config.model.model_name_or_path)

    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name_or_path, **kwargs
    )

    # OPT-based BLIP-2 (blip2-opt-*) needs explicit pad_token for generation
    if model.config.model_type == "opt":
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = processor.tokenizer.pad_token_id or 1

    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        checkpoint_path = Path(args.out_checkpoint)
        if (checkpoint_path / "adapter_config.json").exists():
            lora_path = checkpoint_path
        else:
            lora_path = checkpoint_path / "best_model"

        if lora_path.exists():
            logger.info(f"Loading trained LoRA weights from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            logger.warning(f"Could not find LoRA weights at {lora_path}. Using base model.")
    if not config.model.load_in_8bit and not config.model.load_in_4bit:
        model = model.to(device)

    predictor = Blip2Predictor(model, processor, device)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting BLIP-2 inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        output = predictor.predict(str(image_path), item["question"], item.get("options", []))
        if not output:
            output = "--"
        predictions.append(build_result_record(item, index, output))

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- BLIP-2 Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
