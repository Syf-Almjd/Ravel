"""Ravel's checks as a library: one call, no server, no model call.

    from ravel import check
    verdict = check(user_prompt)                       # direction="input"
    verdict = check(model_reply, direction="output")

On input it runs the sanitizer (invisible characters, look-alike letters, injection markers, size) and the guard
(the keyword blocklist, then the TF-IDF + SVM classifier). On output it runs the blocklist and looks for invisible
characters and chat-template markers; the classifier is reported but never blocks there, because it was trained on
prompts, not on model replies. The same text and policy always give the same verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from . import guard, sanitizer

__all__ = ["Finding", "Policy", "Verdict", "check", "__version__"]
__version__ = "0.1.0"


@dataclass(frozen=True)
class Finding:
    stage: str                      # "sanitizer" | "blocklist" | "classifier"
    code: str                       # "prompt_injection", "invisible_characters", "blocklist", "classifier_unsafe", ...
    detail: str
    category: str | None = None     # the blocklist category, when there is one
    span: tuple[int, int] | None = None  # where in the checked text (after invisible characters are removed)
    blocks: bool = False            # whether this finding blocks under the policy


@dataclass(frozen=True)
class Policy:
    """What runs and what blocks. Every field has a safe default."""

    sanitizer: bool = True
    blocklist: bool = True
    classifier: bool = True
    classifier_threshold: float = 0.50   # block input when P(safe) is below this
    max_bytes: int = 16_384
    max_words: int = 4_096
    block_on_size: bool = True


@dataclass(frozen=True)
class Verdict:
    blocked: bool
    flagged: bool
    reasons: list[Finding] = field(default_factory=list)
    sanitized_text: str = ""            # invisible characters removed, look-alike letters normalised (Ravel's view)
    scores: dict[str, float] = field(default_factory=dict)
    version: str = __version__
    model_sha: str = ""
    ms: float = 0.0


def check(text: str, direction: Literal["input", "output"] = "input", *, policy: Policy | None = None) -> Verdict:
    """Check a prompt (direction="input") or a model's reply (direction="output")."""
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', not {direction!r}")
    p = policy or Policy()
    started = time.perf_counter()
    findings: list[Finding] = []
    scores: dict[str, float] = {}

    cleaned, hidden = sanitizer.strip_invisible(text or "")
    view = sanitizer.normalize_homoglyphs(cleaned).strip()
    if p.sanitizer:
        if hidden:
            findings.append(Finding("sanitizer", "invisible_characters", f"{hidden} invisible character(s) removed"))
        patterns = sanitizer.INJECTION_PATTERNS if direction == "input" else sanitizer.TEMPLATE_MARKERS
        for start, end, matched in sanitizer.find_injections(view, patterns):
            code = "prompt_injection" if direction == "input" else "template_marker"
            findings.append(Finding("sanitizer", code, f"matched {matched!r}", span=(start, end), blocks=True))
        if direction == "input":
            for problem in sanitizer.size_problems(view, p.max_bytes, p.max_words):
                findings.append(Finding("sanitizer", problem.split(":")[0], problem, blocks=p.block_on_size))

    if p.blocklist:
        for rule, start, end, matched in guard.match_blocklist(view):
            findings.append(Finding("blocklist", "blocklist", f"rule {rule.index} matched {matched!r}", category=rule.category, span=(start, end), blocks=True))

    if p.classifier and view:
        safe = guard.safe_probability(view)
        scores["classifier_safe"] = round(safe, 4)
        if safe < p.classifier_threshold:
            findings.append(Finding("classifier", "classifier_unsafe", f"P(safe) = {safe:.3f} < {p.classifier_threshold}",
                                    blocks=direction == "input"))

    blocked = any(f.blocks for f in findings)
    return Verdict(blocked=blocked, flagged=bool(findings) and not blocked, reasons=findings, sanitized_text=view, scores=scores,
                   model_sha=guard.model_sha() if p.classifier else "", ms=round((time.perf_counter() - started) * 1000, 2))
