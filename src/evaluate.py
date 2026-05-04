"""Lightweight illustrative comparison utilities."""

from __future__ import annotations

from collections import Counter
import re

HORROR_KEYWORDS = (
    "shadow",
    "shadows",
    "whisper",
    "whispers",
    "dark",
    "darkness",
    "cold",
    "creak",
    "creaked",
    "silence",
    "silent",
    "empty",
    "flicker",
    "flickered",
    "moon",
    "fog",
    "eerie",
    "uneasy",
    "dread",
    "ominous",
    "door",
    "hallway",
)


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercase words for simple counting."""

    return re.findall(r"[a-zA-Z']+", text.lower())


def word_count(text: str) -> int:
    """Return a simple word count."""

    return len(tokenize_words(text))


def keyword_counts(
    text: str,
    *,
    keywords: tuple[str, ...] = HORROR_KEYWORDS,
) -> Counter[str]:
    """Count exact keyword matches in text."""

    words = tokenize_words(text)
    allowed = set(keywords)
    return Counter(word for word in words if word in allowed)


def analyze_text(text: str) -> dict[str, object]:
    """Return simple descriptive statistics for a generated text."""

    counts = keyword_counts(text)
    return {
        "word_count": word_count(text),
        "horror_keyword_total": sum(counts.values()),
        "horror_keyword_counts": dict(sorted(counts.items())),
    }


def format_comparison(baseline: str, steered: str) -> str:
    """Format a baseline-vs-steered analysis report."""

    baseline_analysis = analyze_text(baseline)
    steered_analysis = analyze_text(steered)

    lines = [
        "=== Illustrative Comparison ===",
        "",
        "This is a lightweight descriptive check, not a rigorous evaluation.",
        "",
        "| Metric | Baseline | Steered |",
        "| --- | ---: | ---: |",
        (
            f"| Word count | {baseline_analysis['word_count']} | "
            f"{steered_analysis['word_count']} |"
        ),
        (
            f"| Horror keyword total | {baseline_analysis['horror_keyword_total']} | "
            f"{steered_analysis['horror_keyword_total']} |"
        ),
        "",
        "Baseline horror keywords:",
        str(baseline_analysis["horror_keyword_counts"]),
        "",
        "Steered horror keywords:",
        str(steered_analysis["horror_keyword_counts"]),
        "",
        "Future metrics could include embedding similarity, human ratings, or a "
        "classifier-based horror intensity score.",
    ]
    return "\n".join(lines)
