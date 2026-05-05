"""Runtime helpers for local developer ergonomics.

This lab is often run on Windows laptops where:
  - Hugging Face / fastembed default caches land on a nearly-full C: drive.
  - Symlink creation is blocked unless Developer Mode or admin privileges are enabled.

We centralize environment setup here so scripts, notebooks, and the API all get
the same stable cache/config behaviour.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_runtime() -> Path:
    """Set cache and console defaults for repeatable local execution."""
    repo_root = Path(__file__).resolve().parent.parent
    cache_root = repo_root / ".cache"
    hf_home = cache_root / "hf"
    fastembed_cache = cache_root / "fastembed"

    for path in (cache_root, hf_home, fastembed_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(fastembed_cache))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(hf_home / "xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except ValueError:
                pass

    return repo_root
