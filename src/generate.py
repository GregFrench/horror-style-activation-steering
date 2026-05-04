"""Generation helpers for baseline and steered stories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.prompts import DEFAULT_SYSTEM_PROMPT, format_chat_prompt
from src.utils import get_input_device, set_seed


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings for text generation."""

    max_new_tokens: int = 400
    temperature: float = 0.8
    top_p: float = 0.9
    seed: int | None = None


def _stop_token_ids(tokenizer: Any) -> list[int]:
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in stop_ids:
        stop_ids.append(eot_id)

    return stop_ids


def generate_response(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    *,
    settings: GenerationSettings | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate only the assistant response text for a story prompt."""

    import torch

    generation_settings = settings or GenerationSettings()
    set_seed(generation_settings.seed)

    prompt_text = format_chat_prompt(
        tokenizer,
        user_prompt,
        system_prompt=system_prompt,
        add_generation_prompt=True,
    )

    device = get_input_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    do_sample = generation_settings.temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": generation_settings.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": _stop_token_ids(tokenizer),
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = generation_settings.temperature
        generation_kwargs["top_p"] = generation_settings.top_p

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[-1]
    new_token_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
