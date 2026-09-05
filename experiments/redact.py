"""Nothing leaves this package with a credential in it.

The experiment store writes configuration, raw trials and reports to disk and
those files are meant to be read, diffed and sometimes committed. A key that
reaches one of them has to be rotated, so the cheapest place to stop it is on
the way out - every write goes through `scrub`.

This is deliberately paranoid rather than clever. It matches key *shapes*, not
a list of known variable names, because the failure it prevents is a key
arriving somewhere nobody thought to look.
"""
from __future__ import annotations

import os
import re
from typing import Any

MASK = "<redacted>"

#: Shapes that are credentials wherever they appear. Ordered longest-first so a
#: specific pattern wins over the generic high-entropy one.
_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),          # Anthropic
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),             # OpenAI-style
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # Exa keys are UUIDs
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),              # Hugging Face
    re.compile(r"\brpa_[A-Za-z0-9_\-]{16,}"),          # Runpod
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),             # GitHub
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),    # Authorization headers
]

#: Any environment variable whose *name* looks like a secret has its *value*
#: scrubbed wherever that value appears verbatim. This catches a key with a
#: shape none of the patterns above knows about.
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)

#: Short values are not treated as secrets: "0", "auto" and "en-us" are real
#: config that happens to live under a name containing KEY.
_MIN_SECRET_LEN = 12


def _live_secrets() -> list[str]:
    out = []
    for name, value in os.environ.items():
        if value and len(value) >= _MIN_SECRET_LEN and _SECRET_NAME.search(name):
            out.append(value)
    # Longest first so a key that contains another key's prefix still fully masks.
    return sorted(set(out), key=len, reverse=True)


def scrub_text(text: str) -> str:
    """Mask every credential-shaped run in `text`."""
    if not text:
        return text
    for secret in _live_secrets():
        text = text.replace(secret, MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    return text


def scrub(value: Any) -> Any:
    """Recursively scrub strings inside dicts, lists and tuples.

    Keys whose *name* looks like a secret are masked outright, even when the
    value's shape is unremarkable - an empty-looking value under `api_key` is
    still not something to write down.
    """
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            # Name-based masking applies to *strings* only. A credential is
            # always text, and matching on the name alone masked every number
            # whose field happened to contain one of these words - which
            # includes `input_tokens`, `output_tokens` and every timing derived
            # from `first_token`. The raw data those runs were meant to
            # preserve came back as "<redacted>".
            if (isinstance(key, str) and _SECRET_NAME.search(key)
                    and isinstance(item, (str, bytes))):
                out[key] = MASK if item else item
            else:
                out[key] = scrub(item)
        return out
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub(v) for v in value)
    return value


def looks_like_secret(text: str) -> bool:
    """True if `text` still contains something credential-shaped.

    The store asserts this is False for everything it has written, so a new
    pattern that slips through fails a test rather than reaching a commit.
    """
    if not text:
        return False
    if any(p.search(text) for p in _PATTERNS):
        return True
    return any(secret in text for secret in _live_secrets())
