"""
Inference script specialized for BLIP-1 (Salesforce/blip-vqa-base).
"""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import (
    build_blip_prompt,
    build_result_record,
    decode_blip_output,
    normalize_blip_answer,
    resolve_test_path,
)
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class BlipPredictor:
    def __init__(self, model, processor, device, finetuned: bool = False):
        self.model = model
        self.processor = processor
        self.device = device
        self.finetuned = finetuned
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: list) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = build_blip_prompt(question, options)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=20)
        decoded = decode_blip_output(self.processor, outputs[0], prompt)

        if self.finetuned:
            decoded = normalize_blip_answer(decoded)
        return decoded


def run_infer(args, config: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building BLIP-1 model and processor for inference...")
    processor = BlipProcessor.from_pretrained(config.model.model_name_or_path)

    kwargs = {"device_map": config.model.device_map}
    if config.model.load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif config.model.load_in_4bit:
        kwargs["load_in_4bit"] = True

    model_path = config.model.model_name_or_path
    finetuned = False
    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        checkpoint_path = Path(args.out_checkpoint) / "best_model"
        if checkpoint_path.exists():
            logger.info(f"Loading fine-tuned BLIP weights from {checkpoint_path}")
            model_path = str(checkpoint_path)
            finetuned = True

    model = BlipForQuestionAnswering.from_pretrained(model_path, **kwargs)
    if not config.model.load_in_8bit and not config.model.load_in_4bit:
        model = model.to(device)

    predictor = BlipPredictor(model, processor, device, finetuned=finetuned)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting BLIP-1 inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        output = predictor.predict(str(image_path), item["question"], item.get("options", []))
        predictions.append(build_result_record(item, index, output))

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- BLIP-1 Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
