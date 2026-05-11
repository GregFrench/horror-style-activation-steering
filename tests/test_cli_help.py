import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_build_steering_vector_help_runs_without_model_dependencies() -> None:
    output = run_help("build_steering_vector.py")

    assert "--pooling" in output
    assert "--include-special-tokens" in output


def test_generate_with_steering_help_runs_without_model_dependencies() -> None:
    output = run_help("generate_with_steering.py")

    assert "--alpha" in output
    assert "--allow-layer-mismatch" in output


def test_compare_outputs_help_runs_without_model_dependencies() -> None:
    output = run_help("compare_outputs.py")

    assert "--baseline-file" in output
    assert "--steered-text" in output
