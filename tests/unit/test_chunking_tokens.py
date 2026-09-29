"""Unit tests for the deterministic model-agnostic lexical token counter.

Infrastructure-free: the counter is a pure function of the text.
"""

from __future__ import annotations

from dynamisrag.chunking.tokens import LEXICAL_TOKEN_PATTERN, count_lexical_tokens


def test_empty_text_has_no_tokens() -> None:
    assert count_lexical_tokens("") == 0


def test_whitespace_only_text_has_no_tokens() -> None:
    assert count_lexical_tokens("   \n\t  ") == 0


def test_words_count_as_single_tokens() -> None:
    assert count_lexical_tokens("hello world") == 2
    assert count_lexical_tokens("one two three four") == 4


def test_numeric_runs_count_as_single_tokens() -> None:
    assert count_lexical_tokens("3.14 2026 42") == 5


def test_standalone_punctuation_counts_per_character() -> None:
    assert count_lexical_tokens("a.b") == 3
    assert count_lexical_tokens("Hello, world!") == 4
    assert count_lexical_tokens("e.g.") == 4


def test_mixed_text_counts_words_and_punctuation() -> None:
    text = "The value is 3.14, see Fig. 2."
    # The value is 3 . 14 , see Fig . 2 .
    assert count_lexical_tokens(text) == 12


def test_whitespace_invariance() -> None:
    assert count_lexical_tokens("hello world") == count_lexical_tokens("hello   world")
    assert count_lexical_tokens("hello world") == count_lexical_tokens(" hello world \n")


def test_unicode_letters_and_punctuation_count_deterministically() -> None:
    assert count_lexical_tokens("café naïve") == 2
    assert count_lexical_tokens("über naïve résumé") == 3
    assert count_lexical_tokens("value: 3.14.") == 6


def test_underscores_are_word_characters() -> None:
    assert LEXICAL_TOKEN_PATTERN.findall("snake_case name") == ["snake_case", "name"]


def test_identical_text_identical_count_repeated() -> None:
    text = "Deterministic scientific chunking: same text, same count."
    assert count_lexical_tokens(text) == count_lexical_tokens(text) == 10
