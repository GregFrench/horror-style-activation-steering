# Horror Style Activation Steering

A PyTorch demo of Llama 3.1 8B activation steering for horror-style story generation.

This project shows how to build a simple activation steering vector from paired writing examples, then apply that vector during autoregressive generation to shift a neutral story prompt toward eerie, PG-13 horror style while preserving the prompt's core subject.

## What This Demonstrates

- Loading `meta-llama/Llama-3.1-8B-Instruct` with Hugging Face Transformers and PyTorch.
- Formatting prompts with the Llama 3.1 Instruct chat template.
- Collecting layer activations from positive and negative style examples.
- Computing a steering vector as:

```text
horror_vector = mean(horror_activations) - mean(neutral_activations)
```

- Registering a clean PyTorch forward hook that adds the vector at a chosen transformer layer during generation.
- Comparing baseline and steered outputs with lightweight, illustrative metrics.

## Why Activation Steering Is Interesting

Activation steering is a small, inspectable intervention on a model's internal representations. Instead of fine-tuning model weights or adding a style instruction to the prompt, we estimate a direction in hidden-state space associated with a target behavior or style, then add that direction while the model generates.

In this demo, the target behavior is not harmful content. It is a PG-13 horror writing style: suspense, atmosphere, ominous imagery, and tension. The goal is to preserve the subject of the original prompt while changing narrative tone.

## Method Overview

1. Collect short positive examples in the target style.
2. Collect short negative examples in a neutral style.
3. Run both sets through the model.
4. Capture hidden states at a configurable transformer layer.
5. Pool each example's activation using either the last token or mean token representation.
6. Compute a vector from neutral style toward horror style.
7. During generation, add `alpha * horror_vector` to the last-token hidden state at that layer.

The core intervention is deliberately small:

```python
hidden_states[:, -1, :] += alpha * steering_vector
```

In practice, layer choice and alpha matter. This repository gives you the machinery to experiment without claiming that one layer or coefficient is universally best.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/your-username/horror-style-activation-steering.git
cd horror-style-activation-steering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Hugging Face Access Note

`meta-llama/Llama-3.1-8B-Instruct` is a gated model on Hugging Face. Before running the model scripts:

1. Request access to the model on Hugging Face.
2. Accept Meta's license terms.
3. Log in locally:

```bash
huggingface-cli login
```

You can also pass a different compatible causal language model with `--model-name`.

## Hardware Note

This repository does not download model weights during validation. Full generation requires a machine with enough memory for Llama 3.1 8B. A GPU with bfloat16 support is recommended. If memory is limited, try `--load-in-4bit` on a CUDA/Linux setup with `bitsandbytes` installed.

## Build a Steering Vector

The included examples are small and meant for demonstration. For stronger steering, add more varied PG-13 positive and neutral examples.

```bash
python scripts/build_steering_vector.py \
  --layer 16 \
  --pooling last_token \
  --max-length 512
```

This writes a file like:

```text
steering_vectors/horror_layer_16.pt
```

The saved payload includes the vector, layer, pooling strategy, model name, and example counts.

## Generate Baseline vs Steered Outputs

```bash
python scripts/generate_with_steering.py \
  --prompt "Write a short story about a person finding an old camera at a garage sale." \
  --layer 16 \
  --alpha 2.0 \
  --max-new-tokens 400
```

Optional 4-bit loading:

```bash
python scripts/generate_with_steering.py \
  --prompt "Write a short story about a student walking home after an evening class." \
  --layer 16 \
  --alpha 2.0 \
  --load-in-4bit
```

Optional saving:

```bash
python scripts/generate_with_steering.py \
  --prompt "Write a short story about a person restoring an old radio." \
  --save-output
```

## Example Output Format

```text
=== Prompt ===
Write a short story about a person finding an old camera at a garage sale.

=== Baseline ===
...

=== Horror Steered ===
...
```

See [examples/sample_outputs.md](examples/sample_outputs.md) for an illustrative, hand-written format example.

## Compare Outputs

The comparison script provides simple descriptive analysis only. It is not a rigorous evaluation of style transfer, semantic preservation, or safety.

```bash
python scripts/compare_outputs.py \
  --baseline-file outputs/baseline.txt \
  --steered-file outputs/steered.txt
```

It reports:

- Word count.
- Counts of simple horror-atmosphere keywords.
- A reminder that proper evaluation would require stronger metrics.

## Project Structure

```text
horror-style-activation-steering/
  README.md
  requirements.txt
  .gitignore
  src/
    __init__.py
    model.py
    steering.py
    prompts.py
    generate.py
    evaluate.py
    utils.py
  examples/
    neutral_prompts.txt
    horror_positive_examples.txt
    neutral_negative_examples.txt
    sample_outputs.md
  scripts/
    build_steering_vector.py
    generate_with_steering.py
    compare_outputs.py
```

## Limitations

- The included example set is intentionally tiny, so results are illustrative.
- Steering can change quality, coherence, or factual consistency if `alpha` is too large.
- The best layer and coefficient are model-dependent.
- Keyword counts are weak proxies for horror style.
- This project does not prove semantic preservation; it only demonstrates a practical steering workflow.
- Hook behavior depends on normal cached autoregressive generation. The default hook avoids steering the full prompt prefill pass and steers generated-token steps.

## Ethical And Safety Note

This project is for style-control experimentation and ML interpretability education. It is scoped to PG-13 suspense and atmosphere, not graphic violence or harmful content. Do not use activation steering to evade safety systems, generate abusive material, or intensify harmful behavior.

## Future Improvements

- Layer sweeps.
- Alpha sweeps.
- Embedding-similarity evaluation for subject preservation.
- Classifier-based horror intensity scoring.
- Human preference evaluation.
- A small web demo.
