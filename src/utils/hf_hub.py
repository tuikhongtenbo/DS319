"""Hugging Face Hub download defaults for large model checkpoints."""

import os


def configure_hf_hub_downloads() -> None:
    """
    Prefer stable HTTP downloads over the XET backend.

    XET parallel reconstruction often fails on large sharded weights with:
    "Internal Writer Error: Background writer channel closed"
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
