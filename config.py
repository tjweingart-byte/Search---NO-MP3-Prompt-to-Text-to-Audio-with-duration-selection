"""Central configuration.

Every value can be overridden with an environment variable so the app can be
tuned without touching code (12-factor style).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import voice_store


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Claude -----------------------------------------------------------
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    # Speed IS the product here: the listener must hear an answer within about
    # a second. Opus with web search took 20-30s to produce its first sentence,
    # which no amount of clever buffering can disguise. Sonnet 5 answers from
    # what it knows almost immediately. MODEL=claude-opus-5 for depth over speed.
    model: str = field(default_factory=lambda: os.environ.get("MODEL", "claude-sonnet-5"))
    max_output_tokens: int = _env_int("MAX_OUTPUT_TOKENS", 16000)
    # HTTP/2 to api.anthropic.com is broken by some proxies and TLS-inspecting
    # middleboxes, which surfaces only as "Connection error". HTTP/1.1 is the
    # default here and is pinned explicitly in anthropic_client.py. Set
    # ANTHROPIC_HTTP2=1 (and `pip install h2`) to opt back in.
    anthropic_http2: bool = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_HTTP2", "0") not in ("0", "false", "False", "")
    )
    # low | medium | high | xhigh | max. Script writing is not a hard reasoning
    # task and effort directly costs time-to-first-audio, so keep it low.
    effort: str = field(default_factory=lambda: os.environ.get("EFFORT", "low"))
    # Padding a short script back to length reintroduces the filler the opener
    # was removed for. Off by default: a briefing that ends when it runs out of
    # substance is better than one stretched to fill the slider.
    allow_topups: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_TOPUPS", "0") not in ("0", "false", "False")
    )
    # OFF by default. Web search is the single biggest cost in time-to-first-word
    # - it front-loads 10-25 seconds before the model writes anything - and the
    # whole product promise is that audio starts almost immediately. Turn it on
    # per request with `search=1`, for a question that genuinely needs today's
    # facts and where the listener will accept waiting for them.
    enable_web_search: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_WEB_SEARCH", "0") not in ("0", "false", "False")
    )
    # Each search adds seconds before the first researched sentence, and the
    # listener hears that wait as preamble. Three is enough for a briefing.
    max_web_searches: int = _env_int("MAX_WEB_SEARCHES", 3)

    # --- Cold open --------------------------------------------------------
    # A small, fast model writes one framing sentence with no tools while the
    # main model is still researching, so speech starts almost immediately.
    # OFF. The opener was prompted to state no facts, which made it filler by
    # construction, and covering a long research wait meant 15-30 seconds of it.
    # Nobody wants that. The interface now shows an honest loading state instead.
    # ENABLE_COLD_OPEN=1 brings it back.
    enable_cold_open: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_COLD_OPEN", "0") not in ("0", "false", "False")
    )
    cold_open_model: str = field(
        default_factory=lambda: os.environ.get("COLD_OPEN_MODEL", "claude-haiku-4-5")
    )
    # Several short framing sentences, released only as long as the main script
    # is still being written. Unused ones are discarded, so this is an upper
    # bound on preamble, not a fixed cost.
    cold_open_words: int = _env_int("COLD_OPEN_WORDS", 70)
    # Longest the opener may keep talking while waiting for the main script.
    # Past this a gap is preferable to endless preamble.
    # Wall-clock ceiling on the opener. Long enough to cover a researched call
    # (web search plus a slow model can run past 30s); past it a gap is better
    # than talking indefinitely about nothing.
    cold_open_max_seconds: float = _env_float("COLD_OPEN_MAX_SECONDS", 60.0)
    # How long to let the main script fail before any opener is spoken, so a
    # bad key produces a clean error rather than an intro to nothing.
    cold_open_grace: float = _env_float("COLD_OPEN_GRACE", 0.35)

    # --- Shared script cache ---------------------------------------------
    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("CACHE_ENABLED", "1") not in ("0", "false", "False")
    )
    cache_backend: str = field(default_factory=lambda: os.environ.get("CACHE_BACKEND", "sqlite"))
    cache_path: str = field(default_factory=lambda: os.environ.get("CACHE_PATH", "scripts.db"))
    # Default lifetime for a cached script.
    cache_ttl_seconds: int = _env_int("CACHE_TTL_SECONDS", 86400)
    # Lifetime for queries that read as time-sensitive ("latest", "today").
    cache_ttl_volatile: int = _env_int("CACHE_TTL_VOLATILE", 900)
    # Use a small model to canonicalise queries before looking them up. Raises
    # the hit rate across differently-worded requests, at the cost of one fast
    # call (~400ms) in front of every request. See cache.canonical_key.
    cache_semantic_key: bool = field(
        default_factory=lambda: os.environ.get("CACHE_SEMANTIC_KEY", "0") not in ("0", "false", "False")
    )
    canonical_key_model: str = field(
        default_factory=lambda: os.environ.get("CANONICAL_KEY_MODEL", "claude-haiku-4-5")
    )

    # --- Duration / pacing ------------------------------------------------
    min_minutes: int = 1
    max_minutes: int = 10
    # Words per minute a natural narrator hits. Used to size the script.
    target_wpm: float = _env_float("TARGET_WPM", 150.0)
    # How far the pacing controller may push the voice to hit the clock.
    min_wpm: float = _env_float("MIN_WPM", 115.0)
    max_wpm: float = _env_float("MAX_WPM", 185.0)
    # Accept anything inside this fraction of the requested length.
    duration_tolerance: float = _env_float("DURATION_TOLERANCE", 0.03)

    # --- Audio ------------------------------------------------------------
    sample_rate: int = _env_int("SAMPLE_RATE", 22050)
    channels: int = 1
    sample_width: int = 2  # 16-bit signed little-endian PCM

    # --- TTS --------------------------------------------------------------
    # auto | piper | espeak | debug
    tts_engine: str = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "auto"))
    # Voice models live in one shared per-user folder (~/.fam/voices by
    # default), NOT inside the project, so a new version of the app finds the
    # voices already downloaded instead of fetching them again. Override with
    # FAM_VOICES_DIR. See voice_store.py.
    voices_dir: str = field(default_factory=lambda: str(voice_store.voices_dir()))
    # Pin one of them as the default; otherwise the first installed is used.
    piper_model: str = field(default_factory=lambda: os.environ.get("PIPER_MODEL", ""))
    espeak_binary: str = field(
        default_factory=lambda: os.environ.get("ESPEAK_BIN", "espeak-ng")
    )
    espeak_voice: str = field(default_factory=lambda: os.environ.get("ESPEAK_VOICE", "en-us"))
    # macOS `say`: present on every Mac, so nothing needs installing there.
    say_binary: str = field(default_factory=lambda: os.environ.get("SAY_BIN", "say"))
    say_voice: str = field(default_factory=lambda: os.environ.get("SAY_VOICE", ""))

    # --- Server -----------------------------------------------------------
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = _env_int("PORT", 8000)
    # Simple abuse guard: seconds between generations from one client.
    rate_limit_seconds: float = _env_float("RATE_LIMIT_SECONDS", 3.0)

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width


settings = Settings()
