#!/usr/bin/env python
"""Build a horror-style activation steering vector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a horror-minus-neutral activation steering vector."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--positive-path",
        type=Path,
        default=REPO_ROOT / "examples" / "horror_positive_examples.txt",
        help="Text file with one PG-13 horror-style snippet per line.",
    )
    parser.add_argument(
        "--negative-path",
        type=Path,
        default=REPO_ROOT / "examples" / "neutral_negative_examples.txt",
        help="Text file with one neutral-style snippet per line.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "steering_vectors")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument(
        "--pooling",
        choices=["last_token", "mean_tokens"],
        default="last_token",
        help="How to pool token activations into one vector per example.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--use-raw-text",
        action="store_true",
        help="Do not wrap examples in the tokenizer chat template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from src.model import load_model_and_tokenizer
    from src.steering import collect_layer_activations, compute_steering_vector
    from src.utils import ensure_dir, read_nonempty_lines

    positive_examples = read_nonempty_lines(args.positive_path)
    negative_examples = read_nonempty_lines(args.negative_path)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    print(f"Collecting {len(positive_examples)} horror-positive activations...")
    positive_activations = collect_layer_activations(
        model,
        tokenizer,
        positive_examples,
        layer=args.layer,
        pooling=args.pooling,
        max_length=args.max_length,
        use_chat_template=not args.use_raw_text,
    )

    print(f"Collecting {len(negative_examples)} neutral-negative activations...")
    negative_activations = collect_layer_activations(
        model,
        tokenizer,
        negative_examples,
        layer=args.layer,
        pooling=args.pooling,
        max_length=args.max_length,
        use_chat_template=not args.use_raw_text,
    )

    # Positive minus negative gives a direction from neutral tone toward the
    # desired horror style. During generation, adding this vector nudges hidden
    # states in that contrastive direction.
    horror_vector = compute_steering_vector(positive_activations, negative_activations)

    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"horror_layer_{args.layer}.pt"
    payload = {
        "vector": horror_vector.cpu(),
        "layer": args.layer,
        "pooling": args.pooling,
        "model_name": args.model_name,
        "positive_path": str(args.positive_path),
        "negative_path": str(args.negative_path),
        "num_positive": len(positive_examples),
        "num_negative": len(negative_examples),
        "use_chat_template": not args.use_raw_text,
    }
    torch.save(payload, output_path)

    print(f"Saved steering vector: {output_path}")
    print(f"Vector shape: {tuple(horror_vector.shape)}")


if __name__ == "__main__":
    main()
