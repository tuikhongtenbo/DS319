"""
Inference script specialized for SpaceLLaVA.
"""

import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import (
    build_result_record,
    build_spacellava_prompt,
    resolve_test_path,
)
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def run_infer(args, config: ExperimentConfig):
    disable_torch_init()

    model_path = config.model.model_name_or_path
    model_name = get_model_name_from_path(model_path)

    logger.info(f"Loading SpaceLLaVA model from {model_path}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=model_name
    )

    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
        if lora_path.exists():
            logger.info(f"Loading trained SpaceLLaVA LoRA weights from {lora_path}")
            model.load_adapter(str(lora_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading SpaceLLaVA test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    predictions = []
    logger.info("Starting SpaceLLaVA inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        prompt_text = build_spacellava_prompt(item["question"], item.get("options", []))
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt_text:
            if model.config.mm_use_im_start_end:
                prompt_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt_text)
            else:
                prompt_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt_text)
        elif model.config.mm_use_im_start_end:
            prompt_text = image_token_se + "\n" + prompt_text
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images([image], image_processor, model.config).to(
            device, dtype=torch.float16
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.1,
            max_new_tokens=20,
            use_cache=True,
        )

        input_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        predictions.append(build_result_record(item, index, output))

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- SpaceLLaVA Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
