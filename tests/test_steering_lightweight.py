import pytest

from src.steering import get_transformer_layers


def test_get_transformer_layers_finds_llama_style_layers() -> None:
    class Inner:
        layers = ["layer-0", "layer-1"]

    class Model:
        model = Inner()

    assert get_transformer_layers(Model()) == ["layer-0", "layer-1"]


def test_get_transformer_layers_recurses_into_base_model() -> None:
    class Inner:
        layers = ["nested-layer"]

    class BaseModel:
        model = Inner()

    class Wrapper:
        base_model = BaseModel()

    assert get_transformer_layers(Wrapper()) == ["nested-layer"]


def test_get_transformer_layers_raises_for_unknown_layout() -> None:
    with pytest.raises(ValueError, match="Could not locate transformer layers"):
        get_transformer_layers(object())
