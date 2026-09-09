"""One vector per query, so two phrasings of the same question can find each
other.

The script cache is keyed on a SHA-256 of the normalised query, which is exact:
either two requests reduce to the identical token set or they are, as far as
the cache is concerned, unrelated. That is cheap, and it is also why the hit
rate is lower than it looks. `normalize_query` does not stem, so "week 5 of the
NFL season" and "NFL week 5" differ by one real word and miss each other
completely, paying ~$0.01 and several seconds to write a script that already
exists.

`cache.canonical_key` was the first answer to that and it charges for it: a
model call (~300-500 ms, ~$0.0002) in front of *every* request, which is pure
overhead on a miss. It is off by default for that reason. A vector is the same
idea moved to write time - embed once when the script is stored, compare
locally when the next question arrives - so the miss path pays microseconds
instead of a round trip.

Two backends, and the difference between them is the honest part:

* **hashing** (default, no dependency) is *lexical*. Signed hashing over tokens
  plus character n-grams, which buys partial overlap and a crude sort of
  stemming ("season"/"seasons") that exact keys cannot express. It does not
  know that "car" and "automobile" are the same thing, and it never will. It is
  here because it needs nothing installed, runs in microseconds, and is
  measurable today.
* **onnx** is a real sentence embedder loaded from `~/.fam/embed`, on the same
  reasoning as the voice models: it ships with the app rather than depending on
  a service that bills per call. **This has not been run on this machine** - no
  model is installed here - so treat the code path as written, not proven,
  exactly like the Piper ONNX path before `verify_voice.py` existed.

Which one is in force is reported by `describe()` and printed by the bench, so
nothing has to guess whether it is looking at semantic or lexical numbers.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import struct
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

#: Width of a hashing-backend vector. 256 keeps a row at 1 KB and a full scan
#: of a few hundred candidates well under a millisecond; the collision rate at
#: this width is irrelevant next to the threshold a match has to clear anyway.
DIMS = 256

#: Character n-gram size. Three, measured: the sweep in
#: `tools/bench_vector_cache.py` found 3 beat 4 and 5 at every weight worth
#: using. Shorter grams share more between related words and, at the low weight
#: below, do not have enough influence to blur unrelated ones together.
_ORDER = 3

#: How much a character n-gram counts next to a whole word. Measured, and much
#: lower than it started: at 0.35 the grams cost recall, and at 1.0 they cost a
#: lot of it, because two unrelated questions about long words start to look
#: similar. At 0.15 they buy two more matches than switching them off entirely
#: and blur nothing. Whole words carry the signal; the grams only soften the
#: edges - "season"/"seasons", "affects"/"effect".
_NGRAM_WEIGHT = 0.15

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")

#: Digits are the detail a vector blurs and a listener notices. "NFL week 5"
#: and "NFL week 6" score high and are not the same episode. Callers use this
#: to refuse a match whose numbers differ - see `cache.comparable`.
_NUMBER = re.compile(r"^\d+$")

#: Numbers people say rather than type. Written out because the bench found the
#: single most dangerous pair in the whole corpus turning on it: "the causes of
#: world war one" and "the causes of world war two" score 0.756, which is under
#: the threshold by four thousandths, and *nothing else* would have stopped
#: them being served each other's episode. The digit guard could not see them
#: because the numbers were spelled. Folding words to digits closes that, and
#: pays for itself again on the other side - "what happened in week five of the
#: nfl" now reaches the episode cached for "NFL week 5", which it could not
#: before.
#:
#: Small numbers only. Beyond twenty people write digits, and every extra entry
#: is another word that stops meaning what it says ("one" as a pronoun already
#: costs a little; "million" would cost more than it saves).
_NUMERALS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}


def tokens(text: str) -> list[str]:
    """Words, lowercased, punctuation gone, spelled numbers folded to digits.

    No filler-word removal here.

    Deliberately not `cache.normalize_query`: that one sorts and de-duplicates
    to build a key, and strips filler words that carry no topic meaning.
    Neither matters to a vector - a bag of hashed tokens is already unordered,
    and a filler word contributes the same small amount to every query, so it
    largely cancels out of a comparison instead of distorting it.
    """
    return [_NUMERALS.get(t, t)
            for t in _SPACE.split(_PUNCT.sub(" ", text.lower())) if t]


def numbers(text: str) -> set[str]:
    """The bare numeric tokens in a query, which a match must agree on."""
    return {t for t in tokens(text) if _NUMBER.match(t)}


def _bucket_and_sign(feature: str) -> tuple[int, float]:
    """Signed hashing: a stable slot, and a stable +/- so collisions cancel
    rather than accumulate."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % DIMS, 1.0 if (value >> 63) & 1 else -1.0


def hashing_vector(text: str) -> list[float]:
    """A unit vector for `text`, from whole words and character n-grams.

    Deterministic across processes and machines - no learned state, no seed -
    which matters because vectors written by one worker are compared by
    another, and a cache that quietly changed its own geometry would serve
    matches that made sense to nobody.
    """
    vec = [0.0] * DIMS
    for word in tokens(text):
        slot, sign = _bucket_and_sign("w:" + word)
        vec[slot] += sign
        padded = "^" + word + "$"
        for i in range(max(1, len(padded) - _ORDER + 1)):
            slot, sign = _bucket_and_sign("g:" + padded[i:i + _ORDER])
            vec[slot] += sign * _NGRAM_WEIGHT
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


# --- The real thing, when someone installs one ----------------------------

def model_dir() -> Path:
    """Where a local sentence embedder lives.

    `~/.fam/embed`, beside `~/.fam/voices` and for the same reason: a model
    downloaded once should be found by every later copy of the app rather than
    re-fetched into a folder that is about to be thrown away.
    """
    override = os.environ.get("FAM_EMBED_MODEL")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fam" / "embed"


class _OnnxEmbedder:
    """A sentence-transformers-style ONNX encoder: tokenise, run, mean-pool.

    Untested on this machine - there is no model here to run it against. It is
    written the way the Piper path was before `verify_voice.py` existed, and it
    deserves the same suspicion until something has actually produced a vector
    with it.
    """

    def __init__(self, directory: Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.dir = directory
        model = directory / "model.onnx"
        vocab = directory / "tokenizer.json"
        for path in (model, vocab):
            if not path.exists():
                raise FileNotFoundError(
                    str(path) + " is missing. FAM_EMBED_BACKEND=onnx needs "
                    "model.onnx and tokenizer.json in " + str(directory) + "."
                )
        self.tokenizer = Tokenizer.from_file(str(vocab))
        self.session = onnxruntime.InferenceSession(
            str(model), providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.dims = int(self.session.get_outputs()[0].shape[-1])

    def encode(self, text: str) -> list[float]:
        import numpy as np

        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(
            None, {k: v for k, v in feed.items() if k in self.inputs}
        )[0]
        weights = mask[..., None].astype(out.dtype)
        pooled = (out * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        vec = pooled[0]
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()


_onnx: Optional[_OnnxEmbedder] = None
_onnx_error = ""


def _load_onnx() -> Optional[_OnnxEmbedder]:
    global _onnx, _onnx_error
    if _onnx is not None or _onnx_error:
        return _onnx
    try:
        _onnx = _OnnxEmbedder(model_dir())
    except Exception as exc:  # ImportError, FileNotFoundError, a bad model
        _onnx_error = type(exc).__name__ + ": " + str(exc)
        # Loud, once. A silent fall back to lexical vectors is exactly the kind
        # of quiet downgrade this project keeps paying for: the numbers would
        # still look fine while measuring something else.
        log.warning(
            "FAM_EMBED_BACKEND=onnx asked for, but no model loaded (%s). "
            "Falling back to the lexical hashing backend - matches will be "
            "lexical, not semantic.", _onnx_error,
        )
    return _onnx


def backend() -> str:
    """"onnx" if a real model is asked for and loads, else "hashing"."""
    if os.environ.get("FAM_EMBED_BACKEND", "hashing").lower() == "onnx":
        if _load_onnx() is not None:
            return "onnx"
    return "hashing"


def dims() -> int:
    return _onnx.dims if backend() == "onnx" and _onnx else DIMS


def describe() -> dict:
    """What is actually in force, for the health report and the bench header."""
    name = backend()
    return {
        "backend": name,
        "dims": dims(),
        # The distinction that decides how far the numbers can be trusted.
        "semantic": name == "onnx",
        "model_dir": str(model_dir()),
        "error": _onnx_error if name != "onnx" else "",
    }


def embed(text: str) -> list[float]:
    """A unit vector for `text`. Never raises - a broken embedder must cost a
    cache miss, not an episode."""
    if backend() == "onnx" and _onnx is not None:
        try:
            return _onnx.encode(text)
        except Exception:
            log.exception("onnx embedding failed; using the lexical vector")
    return hashing_vector(text)


def space() -> str:
    """What distinguishes one vector space from another.

    Part of the cache bucket, so switching backend or width invalidates the old
    vectors instead of comparing coordinates that mean different things.
    """
    return backend() + ":" + str(dims())


# --- Storage and comparison ------------------------------------------------

def pack(vector: list[float]) -> bytes:
    """float32, little-endian. Half the bytes of a double and far more
    precision than a cosine threshold can use."""
    return struct.pack("<" + str(len(vector)) + "f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack("<" + str(len(blob) // 4) + "f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two unit vectors. Returns 0.0 for mismatched widths
    rather than raising - a row from an older vector space is a miss, not an
    error."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
