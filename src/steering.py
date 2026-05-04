"""Activation collection and steering hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from src.prompts import format_style_sample
from src.utils import get_input_device

PoolingStrategy = Literal["last_token", "mean_tokens"]


def _import_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for activation steering. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc


def get_transformer_layers(model: Any) -> Sequence[Any]:
    """Return the transformer block list for common causal LM architectures."""

    candidates = [
        ("model", "layers"),  # LlamaForCausalLM
        ("transformer", "h"),  # GPT-style models
        ("gpt_neox", "layers"),
    ]

    for parent_name, layer_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layer_name, None)
        if layers is not None:
            return layers

    nested_model = getattr(model, "base_model", None)
    if nested_model is not None and nested_model is not model:
        return get_transformer_layers(nested_model)

    raise ValueError(
        "Could not locate transformer layers on the model. Expected a Llama-like "
        "`model.layers` module list or another common causal-LM layout."
    )


def _extract_hidden_states(layer_output: Any) -> torch.Tensor:
    torch = _import_torch()
    if isinstance(layer_output, tuple):
        return layer_output[0]
    if torch.is_tensor(layer_output):
        return layer_output
    if hasattr(layer_output, "last_hidden_state"):
        return layer_output.last_hidden_state
    raise TypeError(f"Unsupported layer output type: {type(layer_output)!r}")


def _replace_hidden_states(layer_output: Any, hidden_states: torch.Tensor) -> Any:
    torch = _import_torch()
    if isinstance(layer_output, tuple):
        return (hidden_states, *layer_output[1:])
    if torch.is_tensor(layer_output):
        return hidden_states
    if hasattr(layer_output, "last_hidden_state"):
        layer_output.last_hidden_state = hidden_states
        return layer_output
    raise TypeError(f"Unsupported layer output type: {type(layer_output)!r}")


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    pooling: PoolingStrategy,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden states with shape [batch, seq, hidden], got {hidden_states.shape}."
        )

    if pooling == "last_token":
        return hidden_states[:, -1, :]
    if pooling == "mean_tokens":
        return hidden_states.mean(dim=1)

    raise ValueError("pooling must be either 'last_token' or 'mean_tokens'.")


class ActivationSteering:
    """Context manager that adds a steering vector at a target transformer layer."""

    def __init__(
        self,
        model: Any,
        steering_vector: torch.Tensor,
        *,
        layer: int,
        alpha: float,
        steer_prompt: bool = False,
    ) -> None:
        self.model = model
        self.layer = layer
        self.alpha = alpha
        self.steer_prompt = steer_prompt
        self._handle: Any | None = None

        vector = steering_vector.detach()
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        if vector.ndim != 1:
            raise ValueError(
                f"Expected steering vector shape [hidden], got {tuple(vector.shape)}."
            )
        self.steering_vector = vector

    def _hook(self, module: Any, args: tuple[Any, ...], output: Any) -> Any:
        torch = _import_torch()
        hidden_states = _extract_hidden_states(output)
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected layer hidden states [batch, seq, hidden], got {hidden_states.shape}."
            )

        hidden_size = hidden_states.shape[-1]
        if hidden_size != self.steering_vector.numel():
            raise ValueError(
                "Steering vector hidden size does not match layer hidden size: "
                f"{self.steering_vector.numel()} != {hidden_size}."
            )

        # During cached generation, the prompt prefill pass usually has seq_len > 1,
        # while later autoregressive steps have seq_len == 1. By default we avoid
        # steering the full prompt pass and steer only generated-token steps.
        if not self.steer_prompt and hidden_states.shape[1] != 1:
            return output

        steered = hidden_states.clone()
        vector = self.steering_vector.to(
            device=steered.device,
            dtype=steered.dtype,
            non_blocking=True,
        )
        steered[:, -1, :] = steered[:, -1, :] + (self.alpha * vector)
        return _replace_hidden_states(output, steered)

    def __enter__(self) -> "ActivationSteering":
        layers = get_transformer_layers(self.model)
        if self.layer < 0 or self.layer >= len(layers):
            raise IndexError(
                f"Layer {self.layer} is out of range for a model with {len(layers)} layers."
            )
        self._handle = layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.remove()

    def remove(self) -> None:
        """Remove the hook if it is currently registered."""

        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def collect_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    layer: int,
    pooling: PoolingStrategy = "last_token",
    max_length: int = 512,
    use_chat_template: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """Collect pooled activations from a transformer layer for a list of texts."""

    torch = _import_torch()

    if not texts:
        raise ValueError("Expected at least one text example.")

    layers = get_transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"Layer {layer} is out of range for {len(layers)} layers.")

    device = get_input_device(model)
    activations: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
        hidden_states = _extract_hidden_states(output)
        pooled = _pool_hidden_states(hidden_states, pooling)
        captured.append(pooled.detach().float().cpu())
        return output

    iterator: Any = texts
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(texts, desc=f"Layer {layer} activations")
        except ImportError:
            iterator = texts

    handle = layers[layer].register_forward_hook(capture_hook)
    try:
        for text in iterator:
            captured.clear()
            formatted = format_style_sample(
                tokenizer,
                text,
                use_chat_template=use_chat_template,
            )
            encoded = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                model(**encoded, use_cache=False)

            if not captured:
                raise RuntimeError(f"Hook did not capture activations for text: {text[:80]!r}")
            activations.append(captured[-1])
    finally:
        handle.remove()

    return torch.cat(activations, dim=0)


def compute_steering_vector(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute a direction from negative examples toward positive examples.

    Positive minus negative is the core contrastive step: adding the resulting
    vector nudges hidden states toward features that are more present in the
    horror examples than in the neutral examples.
    """

    if positive_activations.ndim != 2 or negative_activations.ndim != 2:
        raise ValueError("Expected activation matrices shaped [examples, hidden].")
    if positive_activations.shape[1] != negative_activations.shape[1]:
        raise ValueError(
            "Positive and negative activations must have the same hidden dimension."
        )

    return positive_activations.mean(dim=0) - negative_activations.mean(dim=0)
