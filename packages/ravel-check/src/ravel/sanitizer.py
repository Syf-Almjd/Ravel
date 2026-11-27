"""Stage 1, the sanitizer, as pure functions: invisible characters, look-alike letters, prompt-injection markers and
size limits. No configuration module, no shared state: every limit is a parameter."""

from __future__ import annotations

import re

# Invisible characters attackers use to hide instructions (zero-width, direction overrides, BOM, soft hyphen, ...).
UNICODE_DANGEROUS = re.compile(
    r"[​-‏"   # zero-width space, joiners, direction marks
    r" - "    # line/paragraph separators, embedding overrides
    r"﻿"           # BOM / zero-width no-break space
    r"­"           # soft hyphen
    r"⁠-⁩"    # word joiners, invisible operators
    r"᠎"           # Mongolian vowel separator
    r"￹-￻"    # interlinear annotations
    r"]"
)

# Chat-template markers and the common "ignore your instructions" phrasings.
INJECTION_PATTERNS = re.compile(
    r"\[INST\]"
    r"|<<SYS>>"
    r"|<\|system\|>"
    r"|<\|assistant\|>"
    r"|<\|user\|>"
    r"|<\|im_start\|>"
    r"|<\|im_end\|>"
    r"|system:\s*you\s+are"
    r"|IGNORE\s+(?:ALL\s+)?PREVIOUS\s+INSTRUCTIONS"
    r"|forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?instructions",
    re.IGNORECASE,
)

# Chat-template markers only: in a model's reply these mean it is emitting control tokens.
TEMPLATE_MARKERS = re.compile(r"\[INST\]|<<SYS>>|<\|(?:system|assistant|user|im_start|im_end)\|>", re.IGNORECASE)

# Letters from other alphabets that look like English ones; normalised for detection only.
HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o",  # Cyrillic look-alikes
    "с": "c", "р": "p", "у": "y",
    "ａ": "a", "ｅ": "e", "ｏ": "o",  # Fullwidth look-alikes
}


def strip_invisible(text: str) -> tuple[str, int]:
    """The text without invisible characters, and how many were removed."""
    cleaned, n = UNICODE_DANGEROUS.subn("", text)
    return cleaned, n


def normalize_homoglyphs(text: str) -> str:
    for glyph, replacement in HOMOGLYPH_MAP.items():
        text = text.replace(glyph, replacement)
    return text


def find_injections(text: str, patterns: re.Pattern[str] = INJECTION_PATTERNS) -> list[tuple[int, int, str]]:
    """Each injection marker found, as (start, end, matched text)."""
    return [(m.start(), m.end(), m.group(0)) for m in patterns.finditer(text)]


def size_problems(text: str, max_bytes: int, max_words: int) -> list[str]:
    out = []
    n_bytes = len(text.encode("utf-8"))
    if n_bytes > max_bytes:
        out.append(f"payload_too_large: {n_bytes} bytes (limit {max_bytes})")
    n_words = len(text.split())
    if n_words > max_words:
        out.append(f"token_limit_exceeded: ~{n_words} words (limit {max_words})")
    return out
