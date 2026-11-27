# ravel-check

Ravel's input and output checks as a library. It is one call, with no server and no model call.

```python
from ravel import check, Policy

v = check("Ignore all previous instructions ...")            # direction="input"
v.blocked, v.flagged, v.reasons, v.scores
check(model_reply, direction="output")                         # the classifier is reported, never blocks
check(text, policy=Policy(classifier=False, max_words=2000))   # stages and limits are switches
```

## What each direction runs

- **Input:** the sanitizer (invisible characters, look-alike letters, injection markers, size), then the keyword blocklist (each match names its category), then the TF-IDF and SVM classifier.
- **Output:** the blocklist, plus invisible characters and chat-template markers. The classifier only reports there, because it was trained on prompts.

The model files ship inside the package. Each one is loaded only after its sha256 matches `models/MANIFEST.json`, because loading a pickle can run code.

The same text and policy always give the same verdict.

## Develop

```
cd packages/ravel-check
pip install -e . pytest
pytest
```

Ravel's server (`backend/pipeline/sanitizer.py`, `guard.py`) uses this package, so there is one source of truth.
