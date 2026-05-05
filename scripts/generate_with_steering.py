#!/usr/bin/env python
"""Generate baseline and horror-steered stories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import DEFAULT_MODEL_NAME

DEFAULT_PROMPT = "Write a short story about a student walking home after an evening class."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a baseline story and a horror-steered story."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--vector-path",
        type=Path,
        default=None,
        help="Path to a .pt vector payload. Defaults to steering_vectors/horror_layer_{layer}.pt.",
    )
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--steer-prompt",
        action="store_true",
        help="Also steer the prompt prefill pass. Default steers generated-token steps only.",
    )
    parser.add_argument(
        "--allow-layer-mismatch",
        action="store_true",
        help="Allow applying a saved vector whose recorded layer differs from --layer.",
    )
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.generate import GenerationSettings, generate_response
    from src.model import load_model_and_tokenizer
    from src.steering import ActivationSteering
    from src.utils import (
        ensure_dir,
        load_steering_payload,
        slugify,
        timestamp_slug,
        write_text,
    )

    vector_path = args.vector_path or (
        REPO_ROOT / "steering_vectors" / f"horror_layer_{args.layer}.pt"
    )
    if not vector_path.is_file():
        raise SystemExit(
            f"Steering vector not found: {vector_path}\n"
            "Build one first, for example:\n"
            f"  python scripts/build_steering_vector.py --layer {args.layer}"
        )

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    print("Generating baseline...")
    baseline = generate_response(model, tokenizer, args.prompt, settings=settings)

    print(f"Loading steering vector: {vector_path}")
    steering_payload = load_steering_payload(vector_path)
    if isinstance(steering_payload, dict):
        steering_vector = steering_payload["vector"]
        saved_layer = steering_payload.get("layer")
        if (
            saved_layer is not None
            and int(saved_layer) != args.layer
            and not args.allow_layer_mismatch
        ):
            raise SystemExit(
                f"Vector was built from layer {saved_layer}, but --layer is {args.layer}. "
                "Use the matching layer or pass --allow-layer-mismatch deliberately."
            )
    else:
        steering_vector = steering_payload

    print(f"Generating steered output with layer={args.layer}, alpha={args.alpha}...")
    with ActivationSteering(
        model,
        steering_vector,
        layer=args.layer,
        alpha=args.alpha,
        steer_prompt=args.steer_prompt,
    ):
        steered = generate_response(model, tokenizer, args.prompt, settings=settings)

    output = "\n".join(
        [
            "=== Prompt ===",
            args.prompt,
            "",
            "=== Baseline ===",
            baseline,
            "",
            "=== Horror Steered ===",
            f"(layer={args.layer}, alpha={args.alpha})",
            steered,
            "",
        ]
    )
    print()
    print(output)

    if args.save_output:
        output_dir = ensure_dir(args.output_dir)
        stem = f"{timestamp_slug()}_{slugify(args.prompt)}"
        combined_path = output_dir / f"{stem}.md"
        baseline_path = output_dir / f"{stem}_baseline.txt"
        steered_path = output_dir / f"{stem}_steered.txt"
        write_text(combined_path, output)
        write_text(baseline_path, baseline)
        write_text(steered_path, steered)
        print(f"Saved combined output: {combined_path}")
        print(f"Saved baseline text: {baseline_path}")
        print(f"Saved steered text: {steered_path}")


if __name__ == "__main__":
    main()
