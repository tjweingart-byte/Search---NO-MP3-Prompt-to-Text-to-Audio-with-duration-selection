"""Central configuration.

Every value can be overridden with an environment variable so the app can be
tuned without touching code (12-factor style).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    # claude-opus-5 is the default. Set MODEL=claude-sonnet-5 for a cheaper,
    # lower-latency script pass; the pipeline is model-agnostic.
    model: str = field(default_factory=lambda: os.environ.get("MODEL", "claude-opus-5"))
    max_output_tokens: int = _env_int("MAX_OUTPUT_TOKENS", 16000)
    # low | medium | high | xhigh | max. Script writing is not a hard reasoning
    # task and effort directly costs time-to-first-audio, so keep it low.
    effort: str = field(default_factory=lambda: os.environ.get("EFFORT", "low"))
    # Ground the episode in live sources with Claude's server-side web search.
    enable_web_search: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_WEB_SEARCH", "1") not in ("0", "false", "False")
    )
    max_web_searches: int = _env_int("MAX_WEB_SEARCHES", 5)

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
    piper_binary: str = field(default_factory=lambda: os.environ.get("PIPER_BIN", "piper"))
    piper_model: str = field(default_factory=lambda: os.environ.get("PIPER_MODEL", ""))
    espeak_binary: str = field(
        default_factory=lambda: os.environ.get("ESPEAK_BIN", "espeak-ng")
    )
    espeak_voice: str = field(default_factory=lambda: os.environ.get("ESPEAK_VOICE", "en-us"))

    # --- Server -----------------------------------------------------------
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = _env_int("PORT", 8000)
    # Simple abuse guard: seconds between generations from one client.
    rate_limit_seconds: float = _env_float("RATE_LIMIT_SECONDS", 3.0)

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width


settings = Settings()
