# vLLM Inference

This folder contains vLLM-accelerated inference scripts. vLLM uses **PagedAttention**, **Tensor Parallel**, and **continuous batching** to deliver 2–5x faster inference compared to standard HuggingFace `generate()`.

## Supported Models

Only models whose architectures are natively supported by vLLM are included. Others are deliberately absent to avoid crashes.

| Model | Architecture | vLLM version required | Notes |
|-------|-------------|----------------------|-------|
| **LLaVA** (v1.5, v1.6) | `LlavaForConditionalGeneration` | ≥ 0.4.0 | Most variants on HuggingFace Hub |
| **Qwen2-VL** | `Qwen2VLForConditionalGeneration` | ≥ 0.5.0 | Native multimodal support |

### Unsupported Models (HuggingFace fallback only)

These models are **not** added to the vLLM inference path because vLLM does not support their architecture:

| Model | Reason |
|-------|--------|
| BLIP-2 | Language model (OPT/T5) is not in vLLM's supported list |
| InstructBLIP | Same as BLIP-2 |
| Idefics | Architecture not yet supported by vLLM |
| mPLUG-Owl | Architecture not yet supported by vLLM |
| SpaceLLaVA | Architecture not yet supported by vLLM |
| Qwen (API) | Already calls an external API |

## Installation

```bash
# Optional: only needed when use_vllm: true in config
pip install -r src/requirements/requirement_vllm.txt

# Or install directly:
pip install vllm

# For Qwen2-VL (requires vLLM 0.5+):
pip install "vllm>=0.5.0"
```

## Usage

### 1. Update your config YAML

```yaml
model:
  model_name_or_path: "liuhaotian/llava-v1.5-7b"
  model_type: "llava-v1.5-7b"
  use_vllm: true                    # enable vLLM
  tensor_parallel_size: 1          # number of GPUs (default: 1)
  gpu_memory_utilization: 0.85     # VRAM fraction (default: 0.85)
  max_model_len: 4096              # context length (default: 4096)
```

### 2. Run inference

```bash
python main.py --mode infer --config configs/llava.yaml
```

The dispatcher in `main.py` automatically routes to the vLLM script when `use_vllm: true` and the model is supported. Unsupported models will log a warning and fall back to HuggingFace.

## Files

```
src/inference/
├── inference_vllm_base.py       # Shared utilities (check, engine loader, SamplingParams)
├── inference_vllm_llava.py       # vLLM inference for LLaVA
└── inference_vllm_qwen.py        # vLLM inference for Qwen2-VL
```

## Key Settings

| Config field | Default | Description |
|---|---|---|
| `use_vllm` | `false` | Set `true` to enable vLLM |
| `tensor_parallel_size` | `1` | Number of GPUs for tensor parallel. Use `>1` on multi-GPU systems |
| `gpu_memory_utilization` | `0.85` | Fraction of GPU VRAM used for KV cache |
| `max_model_len` | `4096` | Maximum context length |

## Performance Notes

- **Single GPU**: Set `tensor_parallel_size: 1` (default).
- **Multi-GPU**: Set `tensor_parallel_size: N` where N is the number of GPUs (e.g. 2 or 4).
- **Longer context**: Increase `max_model_len` if you use longer prompts (will use more VRAM).
- **Lower memory**: Reduce `gpu_memory_utilization` if you encounter OOM errors.

## Troubleshooting

**`vLLM not installed`**
```bash
pip install vllm
```

**`vLLM version too old`**
```bash
pip install "vllm>=0.5.0"
```

**`CUDA OOM`**
Reduce `gpu_memory_utilization` (e.g. 0.7) or `max_model_len`.

**`Model architecture not supported`**
If a model type is not in the supported list, the inference script will log a warning and skip. The HuggingFace path is unaffected.

## Adding a New Model to vLLM

When a new architecture becomes supported by vLLM:

1. Add the model key to `VLLM_SUPPORTED_ARCHS` in `inference_vllm_base.py`.
2. Create a new `inference_vllm_<model>.py` script following the pattern of `inference_vllm_llava.py`.
3. Add the routing case in `main.py` `run_infer()` under the `if use_vllm:` block.
