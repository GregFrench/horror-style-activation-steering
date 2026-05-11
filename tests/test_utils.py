import sys
from pathlib import Path

from src import utils


class FakeDevice:
    def __init__(self, spec: str) -> None:
        self.spec = str(spec)
        self.type = self.spec.split(":", maxsplit=1)[0]

    def __repr__(self) -> str:
        return f"FakeDevice({self.spec!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeDevice) and self.spec == other.spec


class FakeTorch:
    device = FakeDevice


class FakeParameter:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device


def test_read_nonempty_lines_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "examples.txt"
    path.write_text("\n# skip me\nfirst\n  second  \n", encoding="utf-8")

    assert utils.read_nonempty_lines(path) == ["first", "second"]


def test_slugify_limits_and_normalizes_text() -> None:
    assert utils.slugify("Old Camera at a Garage Sale!", max_length=15) == (
        "old-camera-at-a"
    )


def test_get_input_device_handles_integer_accelerate_device_map(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    class Model:
        hf_device_map = {"model.embed_tokens": 0}

        def parameters(self):
            return iter([])

    assert utils.get_input_device(Model()) == FakeDevice("cuda:0")


def test_get_input_device_handles_string_integer_device_map(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    class Model:
        hf_device_map = {"model.embed_tokens": "1"}

        def parameters(self):
            return iter([])

    assert utils.get_input_device(Model()) == FakeDevice("cuda:1")


def test_get_input_device_falls_back_to_first_parameter(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    class Model:
        hf_device_map = {"model.embed_tokens": "cpu"}

        def parameters(self):
            return iter([FakeParameter(FakeDevice("cpu"))])

    assert utils.get_input_device(Model()) == FakeDevice("cpu")
