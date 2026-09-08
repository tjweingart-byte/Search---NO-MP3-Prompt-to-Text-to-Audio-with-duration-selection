"""The suite must read the code, never the machine it runs on.

Three failures on real machines had one shape: an environment variable that is
correct for the machine reached a test that was asserting a default. A key in
the shell turned a stubbed endpoint into a real Claude call; the RunPod gate's
`CHATTERBOX_REFERENCE` - which is right, because the pod's reference lives in
the bundle rather than in `~/.fam/voices` - made a test about where the
reference resolves by default fail on the pod and pass in CI.

`tests/conftest.py` removes them all before config.py is imported. This file is
what stops that list falling behind the settings it covers, and what proves the
removal actually happened rather than being described.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import (  # noqa: E402
    FAM_ENVIRONMENT,
    VOICE_ENVIRONMENT,
    config_environment_names,
)


def test_every_setting_config_reads_is_neutralised():
    """A new setting must be added to the list in the same change.

    Without this the list is a snapshot: the next `os.environ.get("SOMETHING")`
    in config.py is a variable a deployment can set and the suite will silently
    obey - which is exactly how the pod's reference reached it.
    """
    covered = set(FAM_ENVIRONMENT) | {"FAM_IGNORE_DOTENV"}
    missing = config_environment_names() - covered
    assert not missing, (
        "config.py reads these and tests/conftest.py does not clear them: "
        f"{sorted(missing)}. Add them to FAM_ENVIRONMENT.")


def test_the_list_has_not_gone_stale_in_the_other_direction():
    """A name left behind after its setting is deleted is harmless but
    misleading - it reads as a setting that exists."""
    known = config_environment_names() | {"FAM_ENV_FILE"}
    stale = set(FAM_ENVIRONMENT) - known
    assert not stale, f"no longer read by config.py: {sorted(stale)}"


def test_the_environment_is_actually_clean_in_this_process():
    """The list is a description; this is the check."""
    present = [name for name in FAM_ENVIRONMENT + VOICE_ENVIRONMENT
               if name in os.environ]
    assert not present, f"still set while the suite runs: {present}"
    assert os.environ.get("FAM_IGNORE_DOTENV") == "1"


def test_the_settings_the_pod_exports_do_not_reach_the_suite():
    """Named individually because these are the ones a validation run sets,
    and each has already caused a failure or is one line from causing one."""
    for name in ("ANTHROPIC_API_KEY", "CHATTERBOX_REFERENCE", "CACHE_ENABLED",
                 "STREAMING_PIPELINE"):
        assert name not in os.environ, f"{name} leaked into the suite"


def test_settings_hold_their_defaults_whatever_the_machine_has_set():
    """The consequence that matters: production's own defaults are what the
    suite tests, on every machine."""
    from config import settings

    assert settings.streaming_pipeline == "legacy"
    assert settings.chatterbox_reference == ""
    assert settings.anthropic_api_key == ""
    assert settings.tts_engine == "auto"
