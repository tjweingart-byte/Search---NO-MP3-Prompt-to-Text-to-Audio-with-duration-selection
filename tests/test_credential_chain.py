"""A credential arrives without anybody typing it, and says where it came from.

`~/.fam/env` (PROBLEMS.md 53) stopped the key being re-pasted on one machine.
The demo does not run on one machine - it runs on a rented pod, a fresh
container, a colleague's laptop - and every one of those is a machine with no
`~/.fam/env` in it. These pin the layer that fixes that: a provider the app can
read for itself, the order it is read in, and the two ways it must fail loudly
rather than quietly.

Nothing here touches a network or a real secrets manager. `cmd:` is a shell
command, so a shell command is exactly what a test can supply.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import credentials  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """Module state is process-wide, so each test starts from nothing."""
    credentials.SOURCES.clear()
    credentials._OWNED.clear()
    credentials._POOL.clear()
    credentials._CURSOR.clear()
    credentials._STATE.update(configured=False, state="unset", detail="", at=0.0,
                              names=[])
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS", "EXA_API_KEY",
                 "EXA_API_KEYS", credentials.PROVIDER_VAR,
                 credentials.TTL_VAR, credentials.TIMEOUT_VAR):
        monkeypatch.delenv(name, raising=False)
    yield
    credentials.SOURCES.clear()
    credentials._OWNED.clear()
    credentials._POOL.clear()
    credentials._CURSOR.clear()


# --- what a secrets manager hands back -----------------------------------

def test_json_output_is_understood():
    """What AWS, Vault and Doppler all emit."""
    found = credentials.parse_payload('{"ANTHROPIC_API_KEY": "sk-ant-1", "EXA_API_KEY": "exa-1"}')
    assert found == {"ANTHROPIC_API_KEY": "sk-ant-1", "EXA_API_KEY": "exa-1"}


def test_dotenv_output_is_understood():
    """What `gcloud secrets versions access` returns if you stored a .env in it."""
    found = credentials.parse_payload('export ANTHROPIC_API_KEY="sk-ant-2"\n# comment\nEXA_API_KEY=exa-2\n')
    assert found == {"ANTHROPIC_API_KEY": "sk-ant-2", "EXA_API_KEY": "exa-2"}


def test_a_nested_value_is_skipped_rather_than_guessed_at():
    """A payload this does not understand must not become a credential that is
    wrong: `{"a": {...}}` stored as the string "{...}" would be sent to the API
    and rejected, and the reason would look like a bad key."""
    found = credentials.parse_payload('{"ANTHROPIC_API_KEY": "sk-ant-3", "meta": {"rotated": 1}}')
    assert found == {"ANTHROPIC_API_KEY": "sk-ant-3"}


def test_output_that_claims_to_be_json_and_is_not_is_refused():
    with pytest.raises(credentials.SecretsUnavailable):
        credentials.parse_payload("{not json at all")


# --- the providers themselves --------------------------------------------

def test_a_command_supplies_the_key(monkeypatch):
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo \'{"ANTHROPIC_API_KEY": "sk-ant-from-cmd"}\'')
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-cmd"
    assert credentials.PROVIDER_VAR in credentials.source("ANTHROPIC_API_KEY")


def test_a_file_supplies_the_key(monkeypatch, tmp_path):
    """The Docker and Kubernetes shape: a secret mounted as a file."""
    blob = tmp_path / "fam.json"
    blob.write_text('{"ANTHROPIC_API_KEY": "sk-ant-from-file"}')
    monkeypatch.setenv(credentials.PROVIDER_VAR, f"file:{blob}")
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"


def test_one_file_can_be_bound_to_one_name(monkeypatch, tmp_path):
    """`/run/secrets/anthropic` holds the key and nothing else - no JSON, no
    name in it. Without the binding form there is nothing to call it."""
    blob = tmp_path / "anthropic"
    blob.write_text("sk-ant-bare\n")
    monkeypatch.setenv(credentials.PROVIDER_VAR, f"file:ANTHROPIC_API_KEY={blob}")
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-bare"


def test_several_providers_are_merged_in_order(monkeypatch):
    """A manager that returns one secret per call needs one line per secret."""
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       "cmd:ANTHROPIC_API_KEY=echo sk-ant-a\n"
                       "cmd:EXA_API_KEY=echo exa-b")
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-a"
    assert os.environ["EXA_API_KEY"] == "exa-b"


# --- the order, which is the whole design --------------------------------

def test_an_explicit_environment_variable_outranks_the_provider(monkeypatch):
    """Someone who exported a key by hand meant it - usually to test one
    specific key against one specific bug. Having it silently replaced by the
    manager is the version of this feature that wastes a day."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-by-hand")
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo \'{"ANTHROPIC_API_KEY": "sk-ant-from-manager"}\'')
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-by-hand"


def test_a_refresh_does_not_overwrite_a_hand_set_key_either(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-by-hand")
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo \'{"ANTHROPIC_API_KEY": "sk-ant-rotated"}\'')
    credentials.load()
    credentials.refresh("test")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-by-hand"


def test_a_rotated_secret_reaches_the_running_process(monkeypatch, tmp_path):
    """This is the claim "rotate the secret, no redeploy" made checkable. The
    manager's answer changes; nothing restarts; the value in force changes."""
    blob = tmp_path / "anthropic"
    blob.write_text("sk-ant-old")
    monkeypatch.setenv(credentials.PROVIDER_VAR, f"file:ANTHROPIC_API_KEY={blob}")
    credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-old"
    blob.write_text("sk-ant-new")
    assert credentials.refresh("rotated") is True
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-new"


# --- the two ways it must fail loudly ------------------------------------

def test_a_broken_provider_raises_rather_than_returning_nothing(monkeypatch):
    """Silently returning {} here means demo mode with no reason given, which
    is the canned script served under a real question all over again (51)."""
    monkeypatch.setenv(credentials.PROVIDER_VAR, "cmd:exit 3")
    with pytest.raises(credentials.SecretsUnavailable):
        credentials.load()
    assert credentials.report()["state"] == "failed"
    assert "exited 3" in credentials.report()["detail"]


def test_a_broken_provider_takes_nothing_from_the_environment(monkeypatch):
    """A failed fetch must not also destroy the key that was already there."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-already-here")
    monkeypatch.setenv(credentials.PROVIDER_VAR, "cmd:exit 3")
    with pytest.raises(credentials.SecretsUnavailable):
        credentials.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-already-here"


def test_a_hanging_provider_does_not_hang_the_server(monkeypatch):
    """Startup blocks on this call. A manager that never answers would
    otherwise be indistinguishable from an app that never starts."""
    monkeypatch.setenv(credentials.TIMEOUT_VAR, "0.3")
    monkeypatch.setenv(credentials.PROVIDER_VAR, "cmd:sleep 5")
    with pytest.raises(credentials.SecretsUnavailable) as exc:
        credentials.load()
    assert "longer than" in str(exc.value)


def test_a_spec_that_is_not_understood_is_refused(monkeypatch):
    """Not treated as a command. `https://vault/...` silently doing nothing is
    a deployment that believes it configured a provider and did not."""
    monkeypatch.setenv(credentials.PROVIDER_VAR, "https://vault.example/fam")
    with pytest.raises(credentials.SecretsUnavailable) as exc:
        credentials.load()
    assert "not understood" in str(exc.value)


# --- the pool -------------------------------------------------------------

def test_one_key_behaves_exactly_as_it_did_before(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-only")
    assert credentials.pool("ANTHROPIC_API_KEY") == ["sk-ant-only"]
    assert credentials.active("ANTHROPIC_API_KEY") == "sk-ant-only"
    assert credentials.demote("ANTHROPIC_API_KEY", "rejected") == ""


def test_a_rejected_key_fails_over_to_the_next(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2")
    assert credentials.active("ANTHROPIC_API_KEY") == "sk-ant-1"
    assert credentials.demote("ANTHROPIC_API_KEY", "spend cap") == "sk-ant-2"


def test_the_key_in_force_is_published_to_the_environment(monkeypatch):
    """The seam the rest of the app hangs off: the Anthropic SDK and
    research.py both read the environment, so a failover reaches them without
    either of them knowing this module exists."""
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2")
    credentials.demote("ANTHROPIC_API_KEY", "rejected")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-2"


def test_an_exhausted_pool_says_so_rather_than_looping(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2")
    credentials.demote("ANTHROPIC_API_KEY", "x")
    credentials.demote("ANTHROPIC_API_KEY", "x")
    assert credentials.active("ANTHROPIC_API_KEY") == ""
    assert credentials.demote("ANTHROPIC_API_KEY", "x") == ""


def test_a_reset_finds_the_keys_behind_the_one_in_force(monkeypatch):
    """`demote` writes the current key into the environment, so rebuilding the
    pool from there would silently shrink it to one."""
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2")
    credentials.demote("ANTHROPIC_API_KEY", "x")
    credentials.reset("ANTHROPIC_API_KEY")
    assert credentials.active("ANTHROPIC_API_KEY") == "sk-ant-1"
    assert credentials.pool("ANTHROPIC_API_KEY") == ["sk-ant-1", "sk-ant-2"]


def test_a_duplicated_key_is_not_a_pool_of_two(monkeypatch):
    """Both variables set to the same key is the likeliest way to configure
    this, and failing over from a key to itself is not failover."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1")
    assert credentials.pool("ANTHROPIC_API_KEY") == ["sk-ant-1"]


# --- the report is safe to paste into a bug ------------------------------

def test_the_report_never_contains_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-secret-1,sk-ant-secret-2")
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo \'{"EXA_API_KEY": "exa-secret"}\'')
    credentials.load()
    blob = repr(credentials.report())
    assert "sk-ant-secret" not in blob
    assert "exa-secret" not in blob
    assert credentials.report()["pools"]["ANTHROPIC_API_KEY"]["keys"] == 2


def test_a_provider_is_named_without_quoting_its_arguments():
    """A secrets command is not a secret, but arguments have carried tokens
    before now, so only the scheme and the program are printed."""
    shown = credentials.describe_spec("cmd:vault kv get -field=data secret/fam --token=hunter2")
    assert "vault" in shown
    assert "hunter2" not in shown


def test_a_pool_only_configuration_still_sends_a_key(monkeypatch):
    """The plural variable is a form only this module reads. Without `prime`,
    a deployment that set ANTHROPIC_API_KEYS and nothing else had two good keys
    in its environment, sent neither, and reported demo mode."""
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "sk-ant-1,sk-ant-2")
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    credentials.prime()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-1"


def test_prime_invents_nothing_when_there_are_no_keys(monkeypatch):
    credentials.prime()
    assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_a_failing_provider_never_republishes_what_it_printed(monkeypatch):
    """`report()` is served by /api/health, which is unauthenticated.

    Two things must not travel that far: a secrets manager's stderr, which
    routinely names account ids, role ARNs, Vault paths and internal hosts; and
    the secret itself, which a wrapper that prints the value and then exits
    non-zero puts on stdout. The classification is public; the diagnostic is
    the server log's.
    """
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo "sk-ant-should-never-be-published"; '
                       'echo "arn:aws:iam::123456789012:role/fam" >&2; exit 3')
    with pytest.raises(credentials.SecretsUnavailable):
        credentials.load()
    published = repr(credentials.report())
    assert "sk-ant-should-never-be-published" not in published, "leaked the secret"
    assert "123456789012" not in published, "leaked the account id"
    # Still says which provider failed and how, or it is not actionable.
    assert "exited 3" in credentials.report()["detail"]


def test_the_operator_still_gets_the_diagnostic_in_the_log(monkeypatch, caplog):
    """Not published is not the same as not reported. An operator who cannot
    see why the provider failed will paste the key somewhere instead."""
    monkeypatch.setenv(credentials.PROVIDER_VAR,
                       'cmd:echo "AccessDeniedException on secret fam" >&2; exit 1')
    with caplog.at_level("ERROR"):
        with pytest.raises(credentials.SecretsUnavailable):
            credentials.load()
    assert "AccessDeniedException" in caplog.text
