"""Unit tests for the deterministic scientific sentence splitter.

Infrastructure-free: the splitter is a pure function of the text and returns
exact character spans into canonical paragraph text.
"""

from __future__ import annotations

from dynamisrag.chunking.sentences import SentenceSpan, split_oversized_sentence, split_sentences


def _spans(text: str) -> list[tuple[int, int, str]]:
    return [(span.start, span.end, span.text) for span in split_sentences(text)]


def test_empty_and_whitespace_text_yields_no_spans() -> None:
    assert split_sentences("") == ()
    assert split_sentences("   \n\t ") == ()


def test_single_sentence_with_terminator() -> None:
    assert _spans("One two.") == [(0, 8, "One two.")]


def test_single_sentence_without_terminator() -> None:
    assert _spans("No terminator here") == [(0, 18, "No terminator here")]


def test_two_simple_sentences() -> None:
    assert _spans("One two. Three four.") == [
        (0, 8, "One two."),
        (9, 20, "Three four."),
    ]


def test_exclamation_and_question_marks_are_terminators() -> None:
    assert _spans("Really? Yes! Go.") == [
        (0, 7, "Really?"),
        (8, 12, "Yes!"),
        (13, 16, "Go."),
    ]


def test_abbreviation_figures_is_not_a_boundary() -> None:
    assert _spans("Fig. 1 shows X. Fig. 2 shows Y.") == [
        (0, 15, "Fig. 1 shows X."),
        (16, 31, "Fig. 2 shows Y."),
    ]


def test_abbreviation_equations_is_not_a_boundary() -> None:
    assert _spans("Eq. (1) holds. Eq. (2) follows.") == [
        (0, 14, "Eq. (1) holds."),
        (15, 31, "Eq. (2) follows."),
    ]


def test_abbreviation_doctor_is_not_a_boundary() -> None:
    assert _spans("Dr. Smith arrived. He left.") == [
        (0, 18, "Dr. Smith arrived."),
        (19, 27, "He left."),
    ]


def test_abbreviation_mister_and_mistress_are_not_boundaries() -> None:
    assert _spans("Mr. and Mrs. Smith arrived. They left.") == [
        (0, 27, "Mr. and Mrs. Smith arrived."),
        (28, 38, "They left."),
    ]


def test_et_al_is_not_a_boundary() -> None:
    assert _spans("Smith et al. studied it. They found X.") == [
        (0, 24, "Smith et al. studied it."),
        (25, 38, "They found X."),
    ]


def test_e_g_is_not_a_boundary() -> None:
    assert _spans("Use e.g. apples. Or oranges.") == [
        (0, 16, "Use e.g. apples."),
        (17, 28, "Or oranges."),
    ]


def test_i_e_is_not_a_boundary() -> None:
    assert _spans("That is i.e. the point. Agreed.") == [
        (0, 23, "That is i.e. the point."),
        (24, 31, "Agreed."),
    ]


def test_vs_is_not_a_boundary() -> None:
    assert _spans("A vs. B was tested. Results followed.") == [
        (0, 19, "A vs. B was tested."),
        (20, 37, "Results followed."),
    ]


def test_numbered_no_is_not_a_boundary() -> None:
    assert _spans("No. 5 was tested. It failed.") == [
        (0, 17, "No. 5 was tested."),
        (18, 28, "It failed."),
    ]


def test_etc_and_approx_are_not_boundaries() -> None:
    assert _spans("Apples, oranges, etc. were sold. Done.") == [
        (0, 32, "Apples, oranges, etc. were sold."),
        (33, 38, "Done."),
    ]
    assert _spans("It costs approx. 5 dollars. Cheap.") == [
        (0, 27, "It costs approx. 5 dollars."),
        (28, 34, "Cheap."),
    ]


def test_decimal_is_not_a_boundary() -> None:
    assert _spans("The value is 3.14. It is set.") == [
        (0, 18, "The value is 3.14."),
        (19, 29, "It is set."),
    ]


def test_version_number_is_not_a_boundary() -> None:
    assert _spans("Version 1.2 is out. Good.") == [
        (0, 19, "Version 1.2 is out."),
        (20, 25, "Good."),
    ]


def test_url_is_not_a_boundary() -> None:
    assert _spans("www.example.com is the site. Visit.") == [
        (0, 28, "www.example.com is the site."),
        (29, 35, "Visit."),
    ]


def test_initials_are_not_a_boundary() -> None:
    assert _spans("A. Smith wrote. B. Jones agreed.") == [
        (0, 15, "A. Smith wrote."),
        (16, 32, "B. Jones agreed."),
    ]


def test_ellipsis_is_one_terminator() -> None:
    assert _spans("Wait... What?") == [
        (0, 7, "Wait..."),
        (8, 13, "What?"),
    ]


def test_period_acronyms_are_not_boundaries() -> None:
    assert _spans("The U.S. is large. Yes.") == [
        (0, 18, "The U.S. is large."),
        (19, 23, "Yes."),
    ]
    assert _spans("The U.K. and U.N. met. They talked.") == [
        (0, 22, "The U.K. and U.N. met."),
        (23, 35, "They talked."),
    ]


def test_pm_abbreviation_can_end_a_sentence() -> None:
    assert _spans("He arrived at 3 p.m. Next.") == [
        (0, 20, "He arrived at 3 p.m."),
        (21, 26, "Next."),
    ]


def test_sentence_ending_punctuation_followed_by_quote() -> None:
    assert _spans('He said "stop." Then left.') == [
        (0, 15, 'He said "stop."'),
        (16, 26, "Then left."),
    ]


def test_sentence_ending_punctuation_followed_by_bracket() -> None:
    assert _spans("The value (see Fig. 1) is high. Next.") == [
        (0, 31, "The value (see Fig. 1) is high."),
        (32, 37, "Next."),
    ]


def test_spans_reconstruct_the_original_text() -> None:
    text = "One two. Three four! Five six? Seven eight."
    spans = split_sentences(text)

    reconstructed = ""
    previous_end = 0
    for span in spans:
        reconstructed += text[previous_end : span.start]
        reconstructed += span.text
        previous_end = span.end
    reconstructed += text[previous_end:]
    assert reconstructed == text


def test_spans_are_exact_substrings_in_order() -> None:
    text = "Alpha beta. Gamma delta."
    spans = split_sentences(text)

    assert [span.text for span in spans] == ["Alpha beta.", "Gamma delta."]
    for span in spans:
        assert text[span.start : span.end] == span.text
    assert spans[0].end <= spans[1].start


def test_oversized_sentence_splits_at_token_boundaries() -> None:
    text = "one two three four five six seven eight nine ten."
    spans = split_oversized_sentence(text, 0, len(text), 4)

    assert [(span.start, span.end, span.text) for span in spans] == [
        (0, 18, "one two three four"),
        (19, 39, "five six seven eight"),
        (40, 49, "nine ten."),
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_oversized_sentence_never_exceeds_max_tokens() -> None:
    from dynamisrag.chunking.tokens import count_lexical_tokens

    text = " ".join(f"word{index}" for index in range(25)) + "."
    spans = split_oversized_sentence(text, 0, len(text), 7)

    assert len(spans) == 4
    for span in spans:
        assert count_lexical_tokens(span.text) <= 7
    assert " ".join(span.text for span in spans) == text


def test_oversized_sentence_with_unicode_tokens() -> None:
    text = "café naïve résumé."
    spans = split_oversized_sentence(text, 0, len(text), 2)

    assert [span.text for span in spans] == ["café naïve", "résumé."]


def test_oversized_sentence_single_token() -> None:
    text = "one."
    spans = split_oversized_sentence(text, 0, len(text), 5)

    assert [(span.start, span.end, span.text) for span in spans] == [(0, 4, "one.")]


def test_oversized_sentence_preserves_leading_whitespace_exclusion() -> None:
    text = "  one two three four five."
    spans = split_oversized_sentence(text, 0, len(text), 2)

    assert spans[0].text == "one two"
    assert text[spans[0].start : spans[0].end] == "one two"


def test_oversized_sentence_keeps_terminal_punctuation_with_its_cluster() -> None:
    """The max boundary lands immediately before the terminal period: the
    period stays with its lexical cluster instead of becoming a
    punctuation-only chunk."""
    text = "one two."
    spans = split_oversized_sentence(text, 0, len(text), 2)

    assert [(span.start, span.end, span.text) for span in spans] == [
        (0, 3, "one"),
        (4, 8, "two."),
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_oversized_sentence_exact_divisibility_before_terminal_punctuation() -> None:
    """Word count is exactly divisible by max before the terminal period:
    the period still stays attached to the last word."""
    text = "one two three four."
    spans = split_oversized_sentence(text, 0, len(text), 4)

    assert [span.text for span in spans] == ["one two three", "four."]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_oversized_sentence_keeps_commas_with_their_cluster() -> None:
    text = "alpha, beta gamma."
    spans = split_oversized_sentence(text, 0, len(text), 3)

    assert [span.text for span in spans] == ["alpha, beta", "gamma."]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_oversized_sentence_keeps_closing_parenthesis_with_its_cluster() -> None:
    text = "alpha (beta) gamma."
    spans = split_oversized_sentence(text, 0, len(text), 4)

    assert [span.text for span in spans] == ["alpha (beta)", "gamma."]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_oversized_sentence_hard_max_holds_when_a_cluster_is_too_long() -> None:
    """A single no-whitespace cluster longer than max_tokens is the one case
    where the hard ceiling forces a split inside a cluster — at token
    boundaries, never inside a code point."""
    from dynamisrag.chunking.tokens import count_lexical_tokens

    text = "a.b.c.d.e.f."
    spans = split_oversized_sentence(text, 0, len(text), 2)

    assert [span.text for span in spans] == ["a.", "b.", "c.", "d.", "e.", "f."]
    for span in spans:
        assert text[span.start : span.end] == span.text
        assert count_lexical_tokens(span.text) <= 2


def test_oversized_sentence_never_emits_punctuation_only_chunks_when_avoidable() -> None:
    text = "one two three four five six seven eight nine ten."
    spans = split_oversized_sentence(text, 0, len(text), 4)

    for span in spans:
        assert any(character.isalnum() for character in span.text), span.text


def test_sentence_span_type_is_exact() -> None:
    span = SentenceSpan(0, 5, "Hello")
    assert span.start == 0 and span.end == 5 and span.text == "Hello"
