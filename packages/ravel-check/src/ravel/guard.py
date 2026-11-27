"""Stage 2, the guard, as a library: the keyword blocklist (with its categories) and the TF-IDF + SVM classifier.

The model files ship inside the package and are read through importlib.resources. Before any pickle is loaded its
sha256 is checked against MANIFEST.json: loading a pickle can run code, so a swapped file is refused, never loaded."""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from dataclasses import dataclass
from importlib import resources

_lock = threading.Lock()
_cache: dict[str, object] = {}


class ModelIntegrityError(RuntimeError):
    """A packaged model file doesn't match its recorded hash."""


@dataclass(frozen=True)
class BlocklistRule:
    index: int
    pattern: re.Pattern[str]
    category: str | None


def _read(package: str, name: str) -> bytes:
    return resources.files(package).joinpath(name).read_bytes()


def blocklist() -> list[BlocklistRule]:
    with _lock:
        if "blocklist" not in _cache:
            data = json.loads(_read("ravel.data", "keyword_blocklist.json"))
            category_of = {i: cat for cat, idx in data.get("categories", {}).items() for i in idx}
            _cache["blocklist"] = [BlocklistRule(i, re.compile(p, re.IGNORECASE), category_of.get(i)) for i, p in enumerate(data.get("patterns", []))]
        return _cache["blocklist"]  # type: ignore[return-value]


def manifest() -> dict:
    return json.loads(_read("ravel.models", "MANIFEST.json"))


def _verified(name: str) -> bytes:
    raw = _read("ravel.models", name)
    want = manifest()["files"].get(name)
    got = hashlib.sha256(raw).hexdigest()
    if want != got:
        raise ModelIntegrityError(f"{name}: sha256 {got} doesn't match the manifest ({want}); refusing to load it")
    return raw


def model():
    """(svm, vectorizer), loaded once and cached, after their hashes are checked."""
    with _lock:
        if "model" not in _cache:
            import joblib

            svm = joblib.load(io.BytesIO(_verified("guard_svm.pkl")))
            vectorizer = joblib.load(io.BytesIO(_verified("tfidf_vectorizer.pkl")))
            _cache["model"] = (svm, vectorizer)
        return _cache["model"]


def model_sha() -> str:
    files = manifest()["files"]
    return hashlib.sha256("".join(files[k] for k in sorted(files)).encode()).hexdigest()[:16]


def match_blocklist(text: str) -> list[tuple[BlocklistRule, int, int, str]]:
    out = []
    for rule in blocklist():
        m = rule.pattern.search(text)
        if m:
            out.append((rule, m.start(), m.end(), m.group(0)))
    return out


def safe_probability(text: str) -> float:
    """The classifier's probability that the text is safe (class 0)."""
    svm, vectorizer = model()
    proba = svm.predict_proba(vectorizer.transform([text]))[0]
    classes = list(getattr(svm, "classes_", [0, 1]))
    return float(proba[classes.index(0) if 0 in classes else 0])
