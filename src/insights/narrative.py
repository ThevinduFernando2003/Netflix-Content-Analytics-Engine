"""Causal language is banned. A hit fails the build."""

from __future__ import annotations

import re
from collections.abc import Iterable


def causal_hits(text: str, stoplist: Iterable[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for verb in stoplist:
        pattern = r"\b" + re.escape(verb.lower()) + r"\b"
        if re.search(pattern, lowered):
            hits.append(verb)
    return hits


def assert_no_causal_language(texts: Iterable[str], stoplist: Iterable[str]) -> None:
    offenders = []
    for text in texts:
        hits = causal_hits(text, stoplist)
        if hits:
            offenders.append((text, hits))
    if offenders:
        raise ValueError(f"Causal language in generated insights: {offenders}")
