from src.prompts import format_chat_prompt, format_style_sample


class RecordingTokenizer:
    chat_template = "fake-template"

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered


class NoTemplateTokenizer:
    chat_template = None


def test_format_chat_prompt_uses_tokenizer_chat_template() -> None:
    tokenizer = RecordingTokenizer()

    prompt = format_chat_prompt(tokenizer, "Write a story about a radio.")

    assert prompt.endswith("assistant:")
    assert tokenizer.calls[0]["tokenize"] is False
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert [message["role"] for message in tokenizer.calls[0]["messages"]] == [
        "system",
        "user",
    ]


def test_format_style_sample_uses_assistant_turn_for_sample_text() -> None:
    tokenizer = RecordingTokenizer()

    rendered = format_style_sample(tokenizer, "The hallway went cold.")

    roles = [message["role"] for message in tokenizer.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "assistant: The hallway went cold." in rendered
    assert tokenizer.calls[0]["add_generation_prompt"] is False


def test_fallback_llama3_template_contains_generation_header() -> None:
    rendered = format_chat_prompt(NoTemplateTokenizer(), "Tell a short story.")

    assert "<|begin_of_text|>" in rendered
    assert "<|start_header_id|>system<|end_header_id|>" in rendered
    assert "<|start_header_id|>user<|end_header_id|>" in rendered
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
