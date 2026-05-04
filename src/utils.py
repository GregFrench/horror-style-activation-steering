"""Small utility helpers shared by scripts."""

from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the repository root based on this file's location."""

    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_nonempty_lines(path: str | Path) -> list[str]:
    """Read non-empty, non-comment lines from a UTF-8 text file."""

    file_path = Path(path)
    lines = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def get_input_device(model: Any) -> Any:
    """Choose a device for input tensors.

    With `device_map="auto"`, Transformers may shard modules. Inputs should
    generally live on the first non-CPU model device.
    """

    import torch

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            device_str = str(device)
            if device_str not in {"cpu", "disk", "meta"}:
                return torch.device(device_str)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def set_seed(seed: int | None) -> None:
    """Set Python, NumPy, and PyTorch seeds when available."""

    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timestamp_slug() -> str:
    """Return a filesystem-friendly timestamp."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text: str, *, max_length: int = 48) -> str:
    """Create a compact filename fragment from text."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "prompt"


def load_steering_vector(path: str | Path) -> Any:
    """Load a steering vector from a `.pt` file.

    The file may contain either a raw tensor or a dictionary with a `vector`
    field, as produced by `scripts/build_steering_vector.py`.
    """

    import torch

    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        if "vector" not in payload:
            raise KeyError(f"Steering payload at {path} does not contain a 'vector' key.")
        return payload["vector"]
    return payload


def write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text, creating parent directories when needed."""

    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(text, encoding="utf-8")
    return file_path
