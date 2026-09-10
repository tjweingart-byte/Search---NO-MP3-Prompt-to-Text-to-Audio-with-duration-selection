"""Central configuration.

Every value can be overridden with an environment variable so the app can be
tuned without touching code (12-factor style).
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

import credentials
import voice_store
from paths import data_path


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
    when you cannot tell which file the app actually read.

    A secrets provider is asked about first because it is the one source that
    is not a file you can go and look at, so it is the one worth naming.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "nowhere - no key is set"
    if "ANTHROPIC_API_KEY" in credentials.SOURCES:
        return credentials.SOURCES["ANTHROPIC_API_KEY"]
    project = pathlib.Path(__file__).resolve().parent / ".env"
    for path, label in ((project, "the project .env"), (shared_env_path(), str(shared_env_path()))):
        try:
            if any(line.strip().lstrip("export ").startswith("ANTHROPIC_API_KEY=")
                   for line in path.read_text().splitlines()):
                return label
        except (OSError, UnicodeDecodeError):
            continue
    return "the environment"


def _dotenv_values() -> dict[str, str]:
    """Read the .env files without applying them.

    Parsed separately from being applied for one reason: `FAM_SECRETS` may
    itself be set in a .env, and the secrets provider has to run *before* the
    files are applied so that a real environment variable still outranks it.
    Reading first and applying after is what lets both be true.

    The shell scripts source .env before starting the server, so for a long
    time nothing in Python needed to. Then `python app.py` - which app.py
    itself offers, in its __main__ block - started the server without it, the
    key was invisible, and the app fell back to the canned demo script while
    .env sat there with a perfectly good key in it. Reading it here means the
    key is found however the app is started.
    """
    # Tests must not change result because of what is in a developer's .env -
    # a key there would flip the app out of demo mode mid-suite. conftest.py
    # sets this before anything imports config.
    if os.environ.get("FAM_IGNORE_DOTENV"):
        return {}
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
        return {}
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
    return found


def _load_dotenv() -> None:
    """Resolve every credential, in the order that needs no human.

        1. the process environment    a platform dashboard, a CI secret, -e
        2. FAM_SECRETS               fetched now, so rotation needs no redeploy
        3. the project .env          a project pinning its own key
        4. ~/.fam/env                the per-machine store (PROBLEMS.md 53)

    A real environment variable always wins: nothing below it overwrites one,
    so `MODEL=... python app.py` still overrides every file and every provider.
    """
    values = _dotenv_values()
    # The provider can be named in a .env - that is how a laptop configures it
    # once - so lift that one variable before asking the provider anything.
    if credentials.PROVIDER_VAR in values and not os.environ.get(credentials.PROVIDER_VAR):
        os.environ[credentials.PROVIDER_VAR] = values[credentials.PROVIDER_VAR]
    # `FAM_IGNORE_DOTENV` silences the provider as well as the files. The name
    # says dotenv, but what conftest.py sets it for is "no ambient credentials
    # in this process", and a suite that shelled out to a developer's secrets
    # manager would be neither hermetic nor fast.
    #
    # A provider that is configured and cannot be read is loud and not fatal.
    # Not fatal, because the .env files below may still hold a usable key and
    # an app that refuses to start has answered a question nobody asked. Loud,
    # because the alternative is falling through to the canned script with no
    # reason given - the silent success this project has lost the most time to.
    if not os.environ.get("FAM_IGNORE_DOTENV"):
        try:
            credentials.load()
        except credentials.SecretsUnavailable:
            pass  # already logged, and reported by health and preflight
    for name, value in values.items():
        if name not in os.environ:
            os.environ[name] = value
    # Everything is now resolved, so the pool has its final contents. Publish
    # its head: `ANTHROPIC_API_KEYS` is a form only `credentials` reads, and a
    # deployment that set only that would otherwise have keys and send none.
    credentials.prime()


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


#: How a researched episode gets its facts.
#:
#: `claude` gives the model Anthropic's server-side `web_search` tool, so it
#: searches while it writes: one call, one credential, and the searching
#: happens inside the model's turn.
#:
#: `exa` retrieves first and hands Claude an evidence packet to read. Two
#: calls and a second credential, but the retrieval is a bounded, timed step
#: this codebase can measure - `research.py` keeps the call and the packet
#: byte-for-byte as the manual benchmark measured them, so the numbers already
#: taken by hand stay comparable.
#:
#: `exa` is the default. That is a real cost: it needs a second credential, and
#: a deployment without EXA_API_KEY cannot research at all - a researched
#: episode will fail rather than quietly search another way. The app says so at
#: startup and on every /api/health, because a missing credential discovered on
#: a listener's first researched question is the shape of failure this project
#: has paid for most.
#:
#: `claude` remains one variable away and needs nothing installed, so a
#: deployment without an Exa key has a working configuration to move to rather
#: than a broken one to endure.
#:
#: A value outside this tuple is refused - at import by
#: `Settings.__post_init__`, and again at retrieval time by `research.retrieve`
#: - rather than falling back to either. A deployment that asked for one and
#: silently got the other would be measuring one thing while believing another.
RESEARCH_BACKENDS = ("claude", "exa")

#: What a deployment gets when it says nothing. Named rather than repeated as a
#: literal, for the same reason as DEFAULT_PIPELINE.
DEFAULT_RESEARCH_BACKEND = "exa"

#: Backends slow enough that the from-knowledge cover earns its second call.
#:
#: `answer_first` exists for exactly one reason: Claude's server-side search
#: costs 10-25 seconds before a word can be written, and a listener will not
#: wait that long in silence. The cover is what fills it - with the durable
#: half of the answer rather than filler, which is the one thing the deleted
#: cold open could never be.
#:
#: Exa is not that. It retrieves in about half a second, and Claude then writes
#: from the packet immediately, so there is no gap to cover. Running the cover
#: anyway costs a second model call, delays the first word, and - the part that
#: matters most - means most of a researched episode is the *unresearched*
#: half: the RunPod run measured the cover speaking 85.8 seconds before
#: research took over. Someone who asked a question that needed today's facts
#: got mostly what the model already knew.
#:
#: So the cover follows the wait rather than the setting. `ANSWER_FIRST=1` or
#: `=0` still wins, because a deployment that has measured its own numbers
#: should not be argued with.
SLOW_RESEARCH_BACKENDS = ("claude",)


def _answer_first_default() -> bool:
    """Cover the wait when there is a wait to cover.

    Reads the environment directly rather than a sibling field: a
    `default_factory` cannot see the rest of the dataclass, and reaching for
    `__post_init__` would overwrite an explicit
    `dataclasses.replace(settings, answer_first=...)`, which several tests and
    `_answer_first` itself depend on.

    The consequence worth knowing: this is derived once, at process start, from
    the environment. `dataclasses.replace(settings, research_backend="claude")`
    does not re-derive it - that is a deliberate in-process override, not a
    deployment being configured.
    """
    raw = os.environ.get("ANSWER_FIRST")
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no", "off")
    backend = os.environ.get(
        "RESEARCH_BACKEND", DEFAULT_RESEARCH_BACKEND).strip().lower()
    return backend in SLOW_RESEARCH_BACKENDS

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
    # the researched half has a sentence ready. Costs a second model call on
    # researched episodes only.
    #
    # **Unset, this now follows the research backend** - see
    # SLOW_RESEARCH_BACKENDS above. It covers a wait, and with Exa retrieving
    # in about half a second there is no wait to cover; on `claude`, where the
    # model's own search costs 10-25 seconds, there is. ANSWER_FIRST=1 or =0
    # still wins outright.
    answer_first: bool = field(default_factory=_answer_first_default)
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
    # claude | exa - see RESEARCH_BACKENDS above. Only consulted when an
    # episode is actually being researched; an unresearched one costs nothing
    # either way.
    research_backend: str = field(
        default_factory=lambda: os.environ.get(
            "RESEARCH_BACKEND", DEFAULT_RESEARCH_BACKEND).strip().lower())
    # The three numbers the manual Exa benchmark hard-coded, which are exactly
    # the knobs worth sweeping. Their defaults reproduce that run: 8 results
    # fetched, the top 3 in the packet, 2 highlights each. Raising
    # `exa_packet_sources` buys more evidence and costs prompt tokens on every
    # researched episode; nobody has measured where that stops paying.
    exa_num_results: int = _env_int("EXA_NUM_RESULTS", 8)
    exa_packet_sources: int = _env_int("EXA_PACKET_SOURCES", 3)
    exa_highlights_per_source: int = _env_int("EXA_HIGHLIGHTS_PER_SOURCE", 2)
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
    # Absolute, and derived from the project root when unset - a bare
    # filename would follow the working directory and quietly open a
    # different, empty cache. See paths.data_path.
    cache_path: str = field(
        default_factory=lambda: data_path("CACHE_PATH", "scripts.db")
    )
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

    # Match a question against *near* neighbours in the cache, not only the
    # identical one. The vector is computed when a script is written, so a
    # lookup costs a local scan (microseconds) rather than the model call
    # CACHE_SEMANTIC_KEY pays on every request. See embeddings.py.
    #
    # Off by default, and the reason is not cost: a false near match plays a
    # confident answer to a question nobody asked, and the shipped embedding
    # backend is lexical rather than semantic (no model is bundled yet), so
    # the thresholds below are tuned against measured pairs and not against
    # meaning. Turn it on once tools/bench_vector_cache.py has been run on
    # traffic that looks like yours.
    cache_vector: bool = field(
        default_factory=lambda: os.environ.get("CACHE_VECTOR", "0") not in ("0", "false", "False")
    )
    # Cosine a near match must clear, and the share of words it must literally
    # share. Both measured, not chosen: tools/bench_vector_cache.py sweeps them
    # against 41 re-phrasings that should collapse and 20 pairs that must not.
    # This is the highest-recall setting at which *every* must-not-collapse
    # pair is refused by a guard rather than by the threshold - so there is no
    # near miss waiting for a query slightly unlike the ones measured.
    cache_vector_threshold: float = _env_float("CACHE_VECTOR_THRESHOLD", 0.68)
    cache_vector_overlap: float = _env_float("CACHE_VECTOR_OVERLAP", 0.6)
    # Rows a near-match scan will look at, newest first.
    cache_vector_scan: int = _env_int("CACHE_VECTOR_SCAN", 400)

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
    # --- Public API -------------------------------------------------------
    # Tier quotas. On by default, because the thing this protects is a GPU and
    # a metered API key reachable by anyone who has the URL, and a ceiling that
    # has to be switched on is a ceiling that is off on the machine nobody
    # checked. `demo.sh` turns it off for a local demo and says so on the way
    # past - judging the writing must not stop after five episodes.
    enforce_quotas: bool = field(
        default_factory=lambda: os.environ.get("ENFORCE_QUOTAS", "1")
        not in ("0", "false", "False", "")
    )
    # Browser origins allowed to call this server, comma separated. Empty means
    # same-origin only, which is what a localhost run and the bundled interface
    # want. A native app is not a browser and sends no Origin, so it needs
    # nothing here - this exists for a web client served from somewhere else.
    #
    # Deliberately not defaulted to "*": with credentialed requests the browser
    # refuses that combination anyway, so a wildcard here would be a setting
    # that looks permissive, is not, and hides the real fix.
    api_origins: str = field(default_factory=lambda: os.environ.get("API_ORIGINS", ""))
    # Audiences accepted from a Google or Apple identity token, comma
    # separated: the iOS bundle id, and any web client id. Empty means that
    # provider is switched off, and `/api/health` says so rather than the
    # sign-in button failing at the point somebody presses it.
    google_client_ids: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLIENT_IDS", ""))
    apple_client_ids: str = field(
        default_factory=lambda: os.environ.get("APPLE_CLIENT_IDS", ""))
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
        if self.research_backend not in RESEARCH_BACKENDS:
            raise ValueError(
                f"RESEARCH_BACKEND={self.research_backend!r} is not a research "
                f"backend. Use one of: {', '.join(RESEARCH_BACKENDS)}."
            )
        for name in ("exa_num_results", "exa_packet_sources",
                     "exa_highlights_per_source"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name.upper()}={getattr(self, name)} must be at least 1. "
                    "Zero would send Claude an empty evidence packet and call "
                    "it research.")
        if self.exa_packet_sources > self.exa_num_results:
            raise ValueError(
                f"EXA_PACKET_SOURCES={self.exa_packet_sources} exceeds "
                f"EXA_NUM_RESULTS={self.exa_num_results}: the packet cannot "
                "hold more sources than were fetched.")

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
    # The key in force outranks the one `settings` captured at import: after a
    # rotation or a failover they are different strings, and the whole point of
    # a fingerprint is to say which one was actually sent.
    key = key or credentials.active("ANTHROPIC_API_KEY") or settings.anthropic_api_key
    if not key:
        return "no key configured"
    shape = "looks like an API key" if key.startswith("sk-ant-") else (
        "DOES NOT start with sk-ant- - is this an API key?")
    return f"{key[:8]}...{key[-4:]} ({len(key)} chars, {shape})"
