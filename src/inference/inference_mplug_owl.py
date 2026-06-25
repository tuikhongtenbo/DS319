"""
Inference script specialized for mPLUG-Owl.
"""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import (
    build_result_record,
    build_spatial_prompt,
    resolve_test_path,
)
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class MplugOwlPredictor:
    def __init__(self, model, processor, tokenizer, device):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: list) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt_text = build_spatial_prompt(question, options)
        prompts = [
            "The following is a conversation between a curious human and AI assistant.\n"
            f"Human: <image>\nHuman: {prompt_text}\nAI: "
        ]

        inputs = self.processor(text=prompts, images=[image], return_tensors="pt")
        inputs = {
            key: value.bfloat16() if value.dtype == torch.float else value
            for key, value in inputs.items()
        }
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_length=512,
            temperature=0.1,
        )
        return self.tokenizer.decode(generated.tolist()[0], skip_special_tokens=True).strip().rstrip(".")


def run_infer(args, config: ExperimentConfig):
    try:
        from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
        from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
        from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
        from peft import PeftModel
    except ImportError as error:
        logger.error(f"Failed to import mPLUG-Owl libraries: {error}")
        logger.warning("Skipping mPLUG-Owl inference run.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_ckpt = config.model.model_name_or_path

    logger.info("Building mPLUG-Owl model and processor for inference...")
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    )

    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
        if lora_path.exists():
            logger.info(f"Loading trained mPLUG-Owl LoRA weights from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))

    model = model.to(device)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    predictor = MplugOwlPredictor(model, processor, tokenizer, device)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading mPLUG-Owl test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting mPLUG-Owl inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        output = predictor.predict(str(image_path), item["question"], item.get("options", []))
        predictions.append(build_result_record(item, index, output))

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- mPLUG-Owl Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
