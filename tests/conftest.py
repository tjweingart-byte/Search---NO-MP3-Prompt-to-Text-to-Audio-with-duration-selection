"""Keep the suite hermetic.

The suite must produce the same result on every machine. Three ways a
developer's or a server's environment used to leak in, each of which turned
"the code changed" into "this machine is different":

1. **`.env` and `~/.fam/env`.** config.py reads both, so that `python app.py`
   finds the key however the server is started. Right for the app, wrong for
   the tests: a key there flips the app out of demo mode mid-suite.

2. **An exported `ANTHROPIC_API_KEY`.** Blocking the files was not enough,
   because anyone who has run the server has the variable in their shell. With
   it set, `/api/script` stops being a stub and makes a real Claude call, and
   `test_generation_is_still_paced` failed on a real machine while passing in
   CI, because a real call takes longer than the pacing window it asserts
   against.

3. **Every other FAM setting.** The RunPod gate exports `CHATTERBOX_REFERENCE`
   before it runs the suite - correctly, since the pod's reference lives in the
   bundle rather than in `~/.fam/voices` - and that made
   `test_the_voice_store_survived_because_chatterbox_uses_it` fail, because it
   asserts where the reference resolves *by default*. The environment was
   right and the test was reading it.

The lesson each time was the same, so this stops fixing them one at a time:
**no environment variable that config.py reads may reach the suite.** A test
that wants a setting sets it explicitly - `dataclasses.replace(settings, ...)`
or `monkeypatch.setenv` plus a reload - which also makes what it is testing
visible in the test rather than in the shell that ran it.

`tests/test_hermetic.py` fails if config.py gains a variable this list does not
cover, so the list cannot quietly fall behind.
"""
import os
import pathlib
import re

import pytest

#: Every environment variable `config.py` reads, removed before it is imported.
#: Kept as a literal list rather than derived at runtime so that adding a
#: setting is a deliberate two-line change and not an invisible one.
FAM_ENVIRONMENT = (
    "ALLOW_TOPUPS", "ANSWER_FIRST", "ANSWER_FIRST_SHARE", "ANTHROPIC_API_KEY",
    "CACHE_BACKEND", "CACHE_ENABLED", "CACHE_PATH", "CACHE_SEMANTIC_KEY",
    "CACHE_TTL_SECONDS", "CACHE_TTL_VOLATILE", "CANONICAL_KEY_MODEL",
    "CHATTERBOX_DEVICE", "CHATTERBOX_REFERENCE", "DURATION_TOLERANCE",
    "ANSWER_FIRST_MAX_SHARE", "EFFORT", "ENABLE_WEB_SEARCH", "ESPEAK_BIN",
    "ESPEAK_VOICE",
    "FAM_ENV_FILE", "HOST", "MAX_OUTPUT_TOKENS", "MAX_WEB_SEARCHES",
    "MAX_WPM", "MIN_WPM", "MODEL", "PORT", "PREROLL_SECONDS",
    "RATE_LIMIT_SECONDS", "READ_LIMIT_PER_WINDOW", "SAMPLE_RATE", "SAY_BIN",
    "SAY_VOICE", "SEARCH_MODE", "STREAMING_PIPELINE", "TARGET_WPM",
    "TTS_ENGINE",
)

#: Where voices live is per-machine state too, and reaches tts.py rather than
#: config.py, so it is not in the list above.
VOICE_ENVIRONMENT = ("FAM_VOICES_DIR", "VOICES_DIR")

#: Set, not cleared: it is what stops config.py reading the two env files.
os.environ["FAM_IGNORE_DOTENV"] = "1"

for name in FAM_ENVIRONMENT + VOICE_ENVIRONMENT:
    os.environ.pop(name, None)


@pytest.fixture(autouse=True)
def hermetic_environment():
    """Clear the settings before every test, and again after it.

    Clearing once at import is not enough, because a test can put them back.
    `tests/test_demo_and_limits.py` exercises `config._load_dotenv()`, whose
    whole job is to write into `os.environ` - so it really does set
    ANTHROPIC_API_KEY and MODEL for the rest of the session, and monkeypatch
    cannot undo a write it did not make. Every test after it then ran with a
    key in the environment.

    Clearing on both sides means no test inherits another's leak and none
    leaves one, whichever order they run in. A test that wants a setting sets
    it inside its own body, where it is visible.
    """
    names = FAM_ENVIRONMENT + VOICE_ENVIRONMENT
    saved = {name: os.environ[name] for name in names if name in os.environ}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(saved)


def config_environment_names() -> set:
    """Every env var config.py reads, found in its source.

    Used by the guard test. Reading the source rather than importing keeps this
    honest about variables read inside a `default_factory` that has not run.
    """
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "config.py").read_text()
    return set(re.findall(r'(?:os\.environ\.(?:get|pop)|_env_int|_env_float|'
                          r'_env_bool)\(\s*"([A-Z_]+)"', source))
