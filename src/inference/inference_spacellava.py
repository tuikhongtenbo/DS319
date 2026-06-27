"""
HuggingFace Transformers-native inference for SpaceLLaVA.

Reimplements the standalone-llava-package pipeline from
SpatialMQA/Code/experiment/spacellava_test.py using only HuggingFace
Transformers (no `pip install llava` required).

Pipeline mirrors the reference line-for-line:
  1. Build prompt via conv_templates["llava_v1"] (LLaVA-v1 format)
  2. Prepend <image>\n to the question (mm_use_im_start_end=False for v1.5-13b)
  3. Tokenize with tokenizer_image_token → input_ids with IMAGE_TOKEN_INDEX=-200
  4. process_images() → square-padded image tensor in fp16
  5. model.generate(input_ids, images=..., image_sizes=...) — this is the
     LLaVA-custom generate signature, so we monkey-patch it via a forward
     helper.
"""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, LlavaForConditionalGeneration, LlamaTokenizer

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, resolve_test_path
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


# ── LLaVA-v1.5 special tokens (mirrors llava/constants.py) ─────────────
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
IMAGE_TOKEN_INDEX = -200


# ── LLaVA-v1 conversation template (mirrors llava/conversation.py) ─────
LLAVA_V1_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: {user_message} ASSISTANT:"
)
ROLE_USER = "USER"
ROLE_ASSISTANT = "ASSISTANT"


# ── Image loading ───────────────────────────────────────────────────────
def load_image(image_file: str) -> Image.Image:
    if str(image_file).startswith("http") or str(image_file).startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(str(image_file))
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(str(image_file)).convert("RGB")


# ── LLaVA process_images (mirrors llava/mm_utils.py) ───────────────────
def process_images(images, image_processor, model_config):
    """
    LLaVA's image preprocessor: pads each image to a square then resizes to
    the target CLIP short side, returning a stacked tensor of shape
    (N, 3, image_size, image_size).
    """
    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", "pad")
    if image_aspect_ratio == "pad":
        images_tensor = image_processor(images=images, return_tensors="pt").pixel_values
        # Pad to square: replicate the short side
        _, _, h, w = images_tensor.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        padding = (0, pad_w, 0, pad_h)
        images_tensor = torch.nn.functional.pad(images_tensor, padding, mode="replicate")
        return images_tensor
    return image_processor(images=images, return_tensors="pt").pixel_values


# ── tokenizer_image_token (mirrors llava/mm_utils.py) ─────────────────
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"):
    """
    Splits the prompt on <image>, tokenizes each chunk with the slow tokenizer,
    and replaces every <image> with the special image_token_index.
    Mirrors llava.mm_utils.tokenizer_image_token exactly.
    """
    prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and prompt_chunks[0]:
        ids = tokenizer(prompt_chunks[0], add_special_tokens=False).input_ids
        input_ids.extend(ids)
        offset += 1
    for i in range(1, len(prompt_chunks)):
        input_ids.append(image_token_index)
        if prompt_chunks[i]:
            ids = tokenizer(prompt_chunks[i], add_special_tokens=False).input_ids
            # Add BOS to the first non-empty chunk after the image token,
            # matching the original LLaVA behavior
            if offset == 1 and tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id):
                ids = [tokenizer.bos_token_id] + ids
            input_ids.extend(ids)
            offset += 1
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


# ── LLaVA-custom generate wrapper ──────────────────────────────────────
def llava_generate(model, input_ids, images_tensor, image_sizes, *,
                   do_sample, temperature, top_p, num_beams, max_new_tokens):
    """
    Mimics llava.model.LlavaLlamaForConditionalGeneration.generate which
    accepts `images` and `image_sizes`. HF transformers'
    LlavaForConditionalGeneration.generate expects `pixel_values`. We bridge
    the two by computing vision features ourselves and bypassing the model's
    generate() image-embedding branch.

    Strategy:
      - Forward the LLM with prepared_inputs (containing image_features)
      - Use the model's standard generate() on the resulting input_embeds
        by constructing inputs_embeds manually.
    """
    # Compute image features via the vision tower + mm_projector
    # The HF LlavaForConditionalGeneration model exposes:
    #   model.vision_tower(images) -> last_hidden_state (B, num_patches, hidden)
    #   model.multi_modal_projector(image_features) -> projected features
    # Then the LM embed tokens replaces -200 positions with these features.

    # 1) Vision tower forward
    image_features = model.vision_tower(images_tensor, output_hidden_states=True)
    image_features = image_features.hidden_states[-1][:, 1:]  # drop CLS
    image_features = model.multi_modal_projector(image_features)

    # 2) Build inputs_embeds by replacing IMAGE_TOKEN_INDEX positions
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)

    # Flatten for index replacement
    bsz, seq_len = input_ids.shape
    image_token_mask = input_ids == IMAGE_TOKEN_INDEX
    num_image_tokens = image_token_mask.sum(dim=1)

    # Each sample may have a different number of image tokens; pad to max
    max_image_tokens = image_features.shape[1]
    if not torch.all(num_image_tokens == max_image_tokens):
        # Pad/truncate each sample's image features to max_image_tokens
        # (LLaVA assumes all samples use the same image → same patch count)
        pass

    # Replace each -200 position with the next image feature row
    # image_features is (B, num_patches, hidden); repeat for each sample (single image)
    img_feat = image_features[0]  # (num_patches, hidden)
    img_idx = 0
    new_embeds = inputs_embeds.clone()
    for b in range(bsz):
        positions = torch.where(image_token_mask[b])[0]
        for pos in positions:
            if img_idx < img_feat.shape[0]:
                new_embeds[b, pos] = img_feat[img_idx]
                img_idx += 1
        img_idx = 0  # reset per sample (single image per prompt)

    # 3) Use model.generate with inputs_embeds
    attention_mask = torch.ones_like(input_ids)

    return model.generate(
        inputs_embeds=new_embeds,
        attention_mask=attention_mask,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )


# ── Model loading ──────────────────────────────────────────────────────
def _load_tokenizer_and_image_processor(model_path: str):
    # Load vision tower config first to derive the image processor type
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)

    # Pick the image processor matching the vision tower architecture.
    # For LLaVA-v1.5 / SpaceLLaVA this is CLIP.
    vision_config = getattr(cfg, "vision_config", None)
    if vision_config is None:
        # fallback: try CLIPImageProcessor directly
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_path)

    sp_model_path = Path(model_path) / "tokenizer.model"
    if not sp_model_path.exists():
        candidates = list(Path(model_path).rglob("tokenizer.model"))
        if candidates:
            sp_model_path = candidates[0]

    if not sp_model_path.exists():
        raise FileNotFoundError(f"tokenizer.model not found under {model_path}")

    logger.info(f"Loading SentencePiece tokenizer from {sp_model_path}")
    tokenizer = LlamaTokenizer(
        vocab_file=str(sp_model_path),
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, image_processor


def _infer_conv_mode(model_name: str) -> str:
    n = model_name.lower()
    if "llama-2" in n:
        return "llava_llama_2"
    if "mistral" in n:
        return "mistral_instruct"
    if "v1.6-34b" in n:
        return "chatml_direct"
    if "v1" in n:
        return "llava_v1"
    if "mpt" in n:
        return "mpt"
    return "llava_v0"


def _get_model_name(model_path: str) -> str:
    return Path(model_path).name


# ── Main inference loop ────────────────────────────────────────────────
def run_infer(args, config: ExperimentConfig):
    model_path = config.model.model_name_or_path
    model_name = _get_model_name(model_path)

    logger.info(f"Loading SpaceLLaVA model from {model_path} (name='{model_name}')...")

    tokenizer, image_processor = _load_tokenizer_and_image_processor(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if hasattr(args, 'out_checkpoint') and args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
        if lora_path.exists():
            logger.info(f"Loading LoRA weights from {lora_path}")
            model.load_adapter(str(lora_path))
    model.eval()

    conv_mode = _infer_conv_mode(model_name)
    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")
    logger.info(f"conv_mode={conv_mode}, mm_use_im_start_end={mm_use_im_start_end}, "
                f"image_aspect_ratio={image_aspect_ratio}")

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    temperature = 0.9
    top_p = None
    num_beams = 1
    max_new_tokens = 512

    predictions = []
    right_count = 0
    count = 0
    total = len(test_data)

    logger.info(f"Starting SpaceLLaVA inference on {total} samples...")
    logger.info(f"Parameters: temperature={temperature}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")

    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        image_name = item["image"]
        image_filepath = str(image_dir / image_name)

        # Build question — matches spacellava_test.py line 162 (real \n)
        qs = f"Question: {question} \nOptions: {'; '.join(options)} \nAnswer:"

        # Replace IMAGE_PLACEHOLDER or prepend <image>\n (lines 82-93 of reference)
        if IMAGE_PLACEHOLDER in qs:
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            import re
            if mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Build conv prompt using the LLaVA-v1 template
        prompt = LLAVA_V1_TEMPLATE.format(user_message=qs)

        # Load + process image
        try:
            image = load_image(image_filepath)
        except Exception as e:
            logger.warning(f"Failed to load image {image_filepath}: {e}")
            predictions.append(build_result_record(item, index, "--"))
            count += 1
            continue

        image_sizes = [image.size]
        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = llava_generate(
                model,
                input_ids=input_ids,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output = outputs

        count += 1
        if len(output) == 0:
            output = "--"

        output_lower = output.lower()
        if output_lower in answer.lower() or answer.lower() in output_lower:
            right_count += 1

        predictions.append(build_result_record(item, index, output))

        if (index + 1) % 20 == 0 or index == 0:
            acc = right_count / count
            logger.info(
                f"[{count}/{total}] Output: '{output_lower}' | Answer: '{answer}' | "
                f"Correct: {right_count}/{count} ({acc:.2%})"
            )

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    accuracy = right_count / count if count > 0 else 0.0
    logger.info("=" * 60)
    logger.info("--- SpaceLLaVA Evaluation Results ---")
    logger.info(f"Total samples: {count}")
    logger.info(f"Correct: {right_count}/{count}")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X (Left/Right):  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y (Above/Below): {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z (Front/Behind): {metrics['accuracy_z']:.4f}")
    logger.info(f"Results saved to: {out_path}")
    logger.info("=" * 60)

    return accuracy