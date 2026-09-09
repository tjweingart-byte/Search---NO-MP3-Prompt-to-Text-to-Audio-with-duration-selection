"""Central configuration.

Every value can be overridden with an environment variable so the app can be
tuned without touching code (12-factor style).
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

import voice_store


def shared_env_path() -> pathlib.Path:
    """The per-machine settings file, alongside the shared voice store.

    `~/.fam/env` is deliberately outside any project folder. A key kept in a
    project `.env` is lost every time the app is unpacked somewhere new, and
    the workaround for that is pasting the key again - into a terminal, into a
    chat, into whatever is to hand. One file per machine, set once.
    """
    override = os.environ.get("FAM_ENV_FILE")
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path.home() / ".fam" / "env" if pathlib.Path.home() else pathlib.Path(".fam-env")


def key_source() -> str:
    """Where the key in force came from. A key that works is not much comfort
    when you cannot tell which file the app actually read."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "nowhere - no key is set"
    project = pathlib.Path(__file__).resolve().parent / ".env"
    for path, label in ((project, "the project .env"), (shared_env_path(), str(shared_env_path()))):
        try:
            if any(line.strip().lstrip("export ").startswith("ANTHROPIC_API_KEY=")
                   for line in path.read_text().splitlines()):
                return label
        except (OSError, UnicodeDecodeError):
            continue
    return "the environment"


def _load_dotenv() -> None:
    """Read .env into the environment, if it is not already there.

    The shell scripts source .env before starting the server, so for a long
    time nothing in Python needed to. Then `python app.py` - which app.py
    itself offers, in its __main__ block - started the server without it, the
    key was invisible, and the app fell back to the canned demo script while
    .env sat there with a perfectly good key in it. Loading it here means the
    key is found however the app is started.

    A real environment variable always wins: this only fills in what is unset,
    so `MODEL=... python app.py` still overrides the file.
    """
    # Tests must not change result because of what is in a developer's .env -
    # a key there would flip the app out of demo mode mid-suite. conftest.py
    # sets this before anything imports config.
    if os.environ.get("FAM_IGNORE_DOTENV"):
        return
    lines: list[str] = []
    # ~/.fam/env first, project .env second, so the project can override the
    # machine-wide setting. The shared file exists for the same reason
    # ~/.fam/voices does: every new copy of the app is a fresh folder with no
    # .env in it, and re-pasting a key into each one is how keys get pasted
    # into the wrong places.
    for path in (shared_env_path(), pathlib.Path(__file__).resolve().parent / ".env"):
        try:
            lines += path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
    if not lines:
        return
    # Last occurrence wins, which is what `source .env` does. A loader that took
    # the first would disagree with the shell scripts about the same file - and
    # a .env that has been appended to twice (an old key, then the corrected
    # one) would authenticate with the wrong one, silently.
    found: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        # Tolerate `export FOO=bar` and quoted values, which is what people
        # actually write in a .env.
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            found[name] = value
    for name, value in found.items():
        if name not in os.environ:
            os.environ[name] = value


_load_dotenv()


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


#: Which generation pipeline a request runs through.
#:
#: `phase6` is production and is `DEFAULT_PIPELINE` below: a character-bounded
#: script buffer between the model reader and the voice, so synthesis falling
#: behind can never stop the reader, and speech-sized chunks after the first.
#: The first chunk keeps its own latency path - the first complete speakable
#: thought, no word floor.
#:
#: `legacy` is the older path: `pipeline._start` pumps every sentence into a
#: bounded queue the synthesiser drains, one sentence per synthesis call, so a
#: slow voice stops the model reading. It is kept, and kept tested, for the
#: baseline half of a comparison run and for the equivalence suite. Nothing
#: selects it automatically.
#:
#: Both stay listed because rolling back must remain one environment variable
#: and a restart. A value outside this tuple is refused - at import by
#: `Settings.__post_init__`, and again at request time by `pipeline._phase6` -
#: rather than falling back to either.
STREAMING_PIPELINES = ("legacy", "phase6")

#: What a deployment gets when it says nothing. Named rather than repeated as a
#: literal, so "the default" is one fact in one place: `Settings`, the health
#: report and the tests all read it from here.
DEFAULT_PIPELINE = "phase6"


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
    # auto | never | always.
    #
    # `auto` reads the question: one that names a moving target - "latest",
    # "today", "score", "breaking" - gets researched and waits for it; one that
    # does not is answered from what the model already knows, immediately.
    # Search front-loads 10-25 seconds before the first word, so paying that on
    # every episode meant paying it mostly for questions that did not need it.
    # A request can still say search=1 or search=0 explicitly and win.
    search_mode: str = field(
        default_factory=lambda: (
            "always" if os.environ.get("ENABLE_WEB_SEARCH", "") in ("1", "true", "True")
            else os.environ.get("SEARCH_MODE", "auto").lower()
        )
    )
    #: Kept so existing callers and the health report still have a boolean to
    #: read; "does this specific episode search" is now a per-question answer.
    enable_web_search: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_WEB_SEARCH", "0") not in ("0", "false", "False", "")
    )
    max_web_searches: int = _env_int("MAX_WEB_SEARCHES", 3)  # a ceiling, not a target
    # Answer first, research underneath. When an episode is going to be
    # researched, run a second call with no tools that starts writing
    # immediately, speak that while the search runs, and hand over the moment
    # the researched half has a sentence ready. The listener never waits, and
    # what covers the wait is the answer rather than filler - which is the one
    # thing the deleted cold open could never be. Costs a second model call on
    # researched episodes only.
    answer_first: bool = field(
        default_factory=lambda: os.environ.get("ANSWER_FIRST", "1") not in ("0", "false", "False")
    )
    # The most of an episode the instant half may speak before it must give way.
    #
    # Without a ceiling this design quietly defeats itself: synthesis runs far
    # faster than research, so the from-knowledge half can finish the entire
    # episode in the time the search takes, and the listener gets an
    # unresearched answer to a question that was researched *because* it needed
    # today's facts. Reserving the rest means the research always gets said.
    answer_first_share: float = _env_float("ANSWER_FIRST_SHARE", 0.5)
    # How far past that ceiling the cover may go when research is *still not
    # ready*, as a share of the episode.
    #
    # The ceiling above is about sharing. Enforced as a deadline it produced
    # the failure this pair exists to balance: the cover stopped, research had
    # nothing yet, and the pipeline blocked on it - silence in the middle of an
    # episode that had already started. Dead air is worse than an over-long
    # opening, and the opening is a real answer rather than filler.
    #
    # So past the ceiling the cover keeps speaking, and this is where that
    # stops: at 0.8 the researched half still gets a fifth of the episode,
    # which is enough for it to be worth having said. Beyond here, covering has
    # stopped buying anything - research would have nothing left to speak into
    # - so the gap is accepted and logged rather than hidden.
    answer_first_max_share: float = _env_float("ANSWER_FIRST_MAX_SHARE", 0.8)
    # legacy | phase6 - see STREAMING_PIPELINES above.
    #
    # **phase6 is production.** It defaulted to `legacy` until the Phase 6 path
    # had been measured through the real interface; that is now done. Validated
    # on an RTX 4090 running this server with Chatterbox and the real prompt:
    # synthesis began before the model finished writing, playback stayed
    # continuous with no gap reaching the listener, the duration ceiling held,
    # and the first chunk was one complete thought with no word floor.
    #
    # `legacy` is kept, and kept tested, for exactly two things: the baseline
    # half of a comparison run (`tools/pod_production_test.sh` measures both on
    # one card), and the equivalence suite that proves the user-visible
    # contract survives the change of execution strategy. Nothing selects it
    # automatically - reaching it takes setting this variable by hand.
    #
    # An unrecognised value is refused at import rather than falling back: a
    # typo that quietly picks a pipeline is exactly the silent-success failure
    # this project has paid for more than once. `pipeline._phase6` refuses one
    # again at request time, so neither gate stands alone.
    streaming_pipeline: str = field(
        default_factory=lambda: os.environ.get(
            "STREAMING_PIPELINE", DEFAULT_PIPELINE).strip().lower()
    )

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
    # auto | espeak | say | debug
    # A **development** override, not a production setting. Production does
    # not choose an engine: there is one production slot
    # (`tts.PRODUCTION_ENGINES`), filled by Chatterbox. This names a
    # development engine for deterministic local tests, and `auto` - the
    # default, and the only value a deployment should ever have - means "the
    # production engine, or a placeholder tone if this machine cannot run it".
    # It cannot name a production engine into existence: nothing here is one.
    tts_engine: str = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "auto"))
    # --- Chatterbox: the production voice --------------------------------
    # Where the model runs. `auto` picks cuda, then mps, and refuses cpu -
    # Chatterbox on a CPU is slower than speech, so an episode would starve.
    chatterbox_device: str = field(
        default_factory=lambda: os.environ.get("CHATTERBOX_DEVICE", "auto"))
    # The recording Chatterbox clones. Per-machine state, never in the repo:
    # it is somebody's voice. Defaults to reference_3.wav in the shared voice
    # folder, and a rights record must sit beside it clearing consent,
    # commercial use and synthetic voice, or the engine reports unavailable.
    chatterbox_reference: str = field(
        default_factory=lambda: os.environ.get("CHATTERBOX_REFERENCE", ""))
    # Per-machine voice state lives in one shared per-user folder
    # (~/.fam/voices by default), NOT inside the project, so a new version of
    # the app finds it already there instead of fetching it again. This is
    # where Chatterbox's reference recording lives. Override with
    # FAM_VOICES_DIR. See voice_store.py.
    voices_dir: str = field(default_factory=lambda: str(voice_store.voices_dir()))
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
    # The cheap endpoints - JSON reads and cache lookups - need a ceiling, not
    # a pace. Opening a tab fires several at once, so anything that throttles
    # a burst throttles correct use. 0 switches it off.
    read_limit_per_window: int = _env_int("READ_LIMIT_PER_WINDOW", 60)
    # How much *audio* must exist before the response starts. A quantity, not
    # a delay: at TARGET_WPM this is 3.75 words, so any ordinary opening
    # sentence satisfies it on the first chunk and it costs nothing. It exists
    # because models stream in bursts, and because it is the last point at
    # which a failed generation can still become an HTTP error rather than a
    # silent empty episode. See app.PREROLL_SECONDS.
    preroll_seconds: float = _env_float("PREROLL_SECONDS", 1.5)

    def __post_init__(self) -> None:
        """Refuse a configuration that names a pipeline that does not exist.

        Deliberately at construction, so it also catches
        `dataclasses.replace(settings, ...)` - which `pipeline._answer_first`
        uses - and not only the environment. The app failing to start is the
        correct outcome: a misconfigured deployment that serves the wrong
        generation path is worse than one that refuses to serve.
        """
        if self.preroll_seconds <= 0:
            # Zero is not "no preroll", it is a broken contract: on `fmt=wav`
            # the 44-byte header alone satisfies a zero gate, the
            # empty-episode guard then fires, and a perfectly good episode
            # comes back as a 502.
            raise ValueError(
                f"PREROLL_SECONDS={self.preroll_seconds} must be greater than "
                "zero. At zero a streamed WAV's header alone satisfies the "
                "gate and every episode is refused as empty."
            )
        if self.streaming_pipeline not in STREAMING_PIPELINES:
            raise ValueError(
                f"STREAMING_PIPELINE={self.streaming_pipeline!r} is not a "
                f"pipeline. Use one of: {', '.join(STREAMING_PIPELINES)}."
            )

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width


settings = Settings()


def describe_key(key: str = "") -> str:
    """A safe fingerprint of the key in force, for error messages.

    "invalid x-api-key" looks the same whichever wrong key produced it, and the
    first question is always whether the one being sent is the one you think.
    Never prints enough to be a secret: a prefix, a length and the last four.
    """
    key = key or settings.anthropic_api_key
    if not key:
        return "no key configured"
    shape = "looks like an API key" if key.startswith("sk-ant-") else (
        "DOES NOT start with sk-ant- - is this an API key?")
    return f"{key[:8]}...{key[-4:]} ({len(key)} chars, {shape})"
