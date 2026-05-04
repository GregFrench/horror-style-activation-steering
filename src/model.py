"""Model loading utilities for the activation steering demo."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def _import_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load the model. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc


def _import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face Transformers is required to load the model. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc


def resolve_torch_dtype(torch_dtype: str | Any | None = "auto") -> Any:
    """Resolve a user dtype option into a torch dtype.

    The default favors bfloat16 on GPUs that support it, float16 on other CUDA
    devices, and float32 on CPU.
    """

    torch = _import_torch()

    if torch_dtype is None or torch_dtype == "auto":
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    if not isinstance(torch_dtype, str):
        return torch_dtype

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[torch_dtype.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported torch dtype '{torch_dtype}'. Expected one of: "
            f"{', '.join(sorted(dtype_map))}, or 'auto'."
        ) from exc


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    load_in_4bit: bool = False,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | Any | None = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer for generation.

    Parameters
    ----------
    model_name:
        Hugging Face model id or local model directory.
    load_in_4bit:
        Enables bitsandbytes 4-bit quantization when installed.
    device_map:
        Passed to Transformers. The default works well with `accelerate`.
    torch_dtype:
        Dtype string, torch dtype, or "auto".
    trust_remote_code:
        Kept explicit for users who choose non-Llama models that require it.
    """

    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_transformers()
    dtype = resolve_torch_dtype(torch_dtype)

    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit:
        if find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "4-bit loading requested, but bitsandbytes is not installed. Install it "
                "with `pip install bitsandbytes` on a supported CUDA/Linux setup, or "
                "rerun without `--load-in-4bit`."
            )

        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer for '{model_name}'. If this is a gated Hugging "
            "Face model such as Llama 3.1, confirm that you have requested access, "
            "accepted the license, and run `huggingface-cli login`."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{model_name}'. Check the model id/path, Hugging Face "
            "access permissions, available disk space, and GPU/CPU memory. For Llama 3.1 "
            "8B on a memory-limited GPU, try `--load-in-4bit`."
        ) from exc

    model.eval()
    return model, tokenizer
