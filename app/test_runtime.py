from __future__ import annotations

import os
from pathlib import Path

from app.runtime import configure_runtime


def test_configure_runtime_sets_local_cache_dirs() -> None:
    repo_root = configure_runtime()

    assert isinstance(repo_root, Path)
    assert repo_root.name == "Day19-Track2-VectorFeatureStore-Lab"
    assert Path(os.environ["FASTEMBED_CACHE_PATH"]).is_dir()
    assert Path(os.environ["HF_HOME"]).is_dir()
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
