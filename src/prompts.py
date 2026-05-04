"""Prompt formatting helpers for Llama 3.1 Instruct."""

from __future__ import annotations

from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a creative writing assistant. Write coherent, age-appropriate short "
    "fiction that preserves the user's requested subject."
)

STYLE_SAMPLE_SYSTEM_PROMPT = (
    "The following is a PG-13 writing sample. Represent its narrative style and tone."
)


def build_story_messages(
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build chat messages for baseline and steered story generation."""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def _fallback_llama3_chat_template(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Fallback formatter for Llama 3 style chat tokens.

    Most users should get the official template from the tokenizer. This fallback
    keeps scripts usable with compatible tokenizers that do not expose one.
    """

    chunks = ["<|begin_of_text|>"]
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        chunks.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    if add_generation_prompt:
        chunks.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(chunks)


def format_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
) -> str:
    """Format a user prompt using the tokenizer's chat template when available."""

    messages = build_story_messages(user_prompt, system_prompt=system_prompt)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    return _fallback_llama3_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
    )


def format_style_sample(
    tokenizer: Any,
    sample_text: str,
    *,
    use_chat_template: bool = True,
) -> str:
    """Format a style example for activation collection.

    The sample is represented as assistant text because the vector is meant to
    capture a writing style the model can produce, not an instruction to follow.
    A short user turn keeps the conversation shape compatible with strict chat
    templates.
    """

    sample = sample_text.strip()
    if not use_chat_template:
        return sample

    messages = [
        {"role": "system", "content": STYLE_SAMPLE_SYSTEM_PROMPT},
        {"role": "user", "content": "Read this writing sample."},
        {"role": "assistant", "content": sample},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return _fallback_llama3_chat_template(messages, add_generation_prompt=False)
