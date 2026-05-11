from src.evaluate import analyze_text, format_comparison, keyword_counts, word_count


def test_word_count_handles_basic_punctuation() -> None:
    assert word_count("The hallway was quiet, then it wasn't.") == 7


def test_keyword_counts_are_case_insensitive() -> None:
    counts = keyword_counts("The Shadows held a cold, silent hallway.")

    assert counts["shadows"] == 1
    assert counts["cold"] == 1
    assert counts["silent"] == 1
    assert counts["hallway"] == 1


def test_analyze_text_returns_expected_summary_keys() -> None:
    analysis = analyze_text("A dark hallway stayed silent.")

    assert analysis["word_count"] == 5
    assert analysis["horror_keyword_total"] == 3
    assert analysis["horror_keyword_counts"] == {
        "dark": 1,
        "hallway": 1,
        "silent": 1,
    }


def test_format_comparison_includes_not_rigorous_note() -> None:
    report = format_comparison(
        "Mara walked home under streetlights.",
        "Mara walked home as shadows gathered.",
    )

    assert "not a rigorous evaluation" in report
    assert "| Word count |" in report
    assert "Steered horror keywords" in report
