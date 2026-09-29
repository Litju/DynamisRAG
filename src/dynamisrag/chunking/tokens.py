"""Deterministic model-agnostic lexical token counting (RES-134).

``Passage.token_count`` records this chunker's deterministic sizing count —
not a claim about any future embedding-model tokenization. The counter is a
pure function of the text: identical text on Windows or Linux under the
pinned Python produces the identical count, and no embedding-model
vocabulary is involved anywhere in canonical chunk identity.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["LEXICAL_TOKEN_PATTERN", "count_lexical_tokens"]

LEXICAL_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+|[^\w\s]", re.UNICODE)
"""One token per word/numeric run plus one per standalone punctuation mark.

``\\w+`` captures runs of Unicode word characters (letters, digits and
underscore) as single tokens — words and numeric runs — while ``[^\\w\\s]``
captures each remaining non-space character (standalone punctuation) as its
own token. Whitespace never contributes tokens, so the count is invariant to
whitespace runs.
"""


def count_lexical_tokens(text: str) -> int:
    """Count ``unicode-lexical-v1`` tokens in ``text``.

    Deterministic and total: the same text always yields the same count,
    independent of platform, locale or any embedding-model tokenizer.
    """
    return len(LEXICAL_TOKEN_PATTERN.findall(text))
