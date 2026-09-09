"""Where a credential comes from, when nobody is there to type it.

`~/.fam/env` (PROBLEMS.md 53) stopped the key being re-pasted *on one machine*.
It does nothing for the case that actually keeps happening: the demo runs on a
**fresh machine** - a rented GPU pod, a new container, a colleague's laptop, a
CI runner - and a fresh machine has no `~/.fam/env` in it. So the key gets typed
again, and the thing §53 diagnosed ("nobody re-enters a credential four times
because they enjoy it") repeats one layer up.

The fix is the same shape as §53's, moved one level out: put the credential
somewhere the *deployment* can reach programmatically, and let the app fetch it
at the moment it needs it.

    FAM_SECRETS='cmd:doppler secrets download --no-file --format json'

That is one **non-secret** variable on the host template. It replaces pasting
every secret into every host template, and it is the only line that has to be
set on a machine that has never run FAM before.

## The chain, highest priority first

    1. the process environment    a platform dashboard, a CI secret, `docker -e`
    2. FAM_SECRETS               fetched at runtime, so rotation needs no redeploy
    3. the project .env          a project pinning its own key
    4. ~/.fam/env                the per-machine store from §53

An explicit environment variable still wins, because a person who set one meant
it. Everything below it is a way of not having to.

## The provider is a command, deliberately

There is no AWS SDK here, no Vault client, no Doppler library - one hook that
shells out to whatever the deployment already has:

    file:/run/secrets/fam                 Docker / Kubernetes secret files
    file:ANTHROPIC_API_KEY=/run/secrets/anthropic     one file, one value
    cmd:aws secretsmanager get-secret-value --secret-id fam --query SecretString --output text
    cmd:gcloud secrets versions access latest --secret=fam
    cmd:vault kv get -format=json -field=data secret/fam
    cmd:ANTHROPIC_API_KEY=op read op://vault/fam/credential

Every one of those authenticates as the *machine* - an IAM role, a service
account, a workload identity - not as a stored password. No new dependency, no
vendor picked on the app's behalf, and the escape hatch for a manager nobody
has thought of yet is that it is already a shell command.

Output is parsed as a JSON object or as dotenv lines, which between them is
what every one of those commands emits. `NAME=` in front binds the whole
(stripped) output to one variable, for the managers that return a bare secret.
A command that genuinely needs a leading assignment can say `cmd:env FOO=bar ...`.

## Rotation, and the pool

`refresh()` re-runs the provider and replaces the values it owns, so rotating a
secret is "change it in the manager" - no redeploy, no restart. It is called on
startup, on the TTL, and when Claude rejects the key in force, which is the
moment the app finds out a rotation happened.

A pool is several keys under `ANTHROPIC_API_KEYS` / `EXA_API_KEYS`. Be precise
about what it buys, because the obvious assumption is wrong:

* **Anthropic rate limits are per organisation, not per key.** A second key from
  the same org shares one bucket and buys **no headroom at all**. What a pool
  buys here is *failover*: a key that is revoked, expired or spend-capped mid
  demo steps aside for the next one instead of ending the demo.
* **Exa's limits are per key** (10 rps on /search). There a pool is real
  headroom.

Anything that claims otherwise is claiming the thing this file exists to stop.

## Failure is loud

If `FAM_SECRETS` is set and the fetch fails, that is said - at startup, in
`/api/health`, and in `tools/demo_preflight.py`. Falling through to "no key, so
the canned script" without naming the reason is the silent success this project
has lost the most time to.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shlex
import subprocess
import time

log = logging.getLogger("credentials")

#: The one variable a fresh machine has to be told. Not itself a secret - it
#: names *where* the secrets are, which is why it is safe on a host template,
#: in a Dockerfile, or in this repository's documentation.
PROVIDER_VAR = "FAM_SECRETS"

#: How long a provider command may take before it is given up on. A secrets
#: manager that hangs must not hang the server: startup blocks on this.
TIMEOUT_VAR = "FAM_SECRETS_TIMEOUT"
DEFAULT_TIMEOUT = 10.0

#: How long a fetched value is trusted before the provider is asked again.
#: Zero means "only when something asks", which is startup plus a rejection.
TTL_VAR = "FAM_SECRETS_TTL"
DEFAULT_TTL = 0.0

#: Credentials this module manages. A pool is the same name with an S.
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "EXA_API_KEY")

#: A variable name, for telling `NAME=command` apart from a command that
#: happens to begin with an assignment. Uppercase only, which every credential
#: variable in this project is and `FOO=bar` in `FOO=bar ./thing` usually is
#: not - and `cmd:env FOO=bar ./thing` is the way to say the other thing.
_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SecretsUnavailable(RuntimeError):
    """A configured provider could not be read. Never swallowed."""


#: Where each managed name came from, for `key_source()` and health. Values are
#: descriptions, never secrets.
SOURCES: dict[str, str] = {}

#: Names this module put into the environment. Only these may be replaced on a
#: refresh: a variable the operator set by hand outranks the provider, and a
#: rotation must not quietly overwrite a deliberate override.
_OWNED: set[str] = set()

#: Last provider outcome, for the report. "unset" until something asks.
_STATE: dict[str, object] = {
    "configured": False, "state": "unset", "detail": "", "at": 0.0, "names": [],
}

#: Per-credential rotation state: the pool, and how far down it we are.
_POOL: dict[str, list[str]] = {}
_CURSOR: dict[str, int] = {}


def _timeout() -> float:
    try:
        return float(os.environ.get(TIMEOUT_VAR, DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _ttl() -> float:
    try:
        return float(os.environ.get(TTL_VAR, DEFAULT_TTL))
    except (TypeError, ValueError):
        return DEFAULT_TTL


def provider_specs() -> list[str]:
    """The configured providers, one per line, in the order they are merged.

    One is the normal case: a single command that returns every secret. Several
    lines are for managers that hand back one secret per call, and a later line
    overrides an earlier one - which is what sourcing two files does.
    """
    raw = os.environ.get(PROVIDER_VAR, "") or ""
    specs = [line.strip() for line in raw.splitlines()]
    return [s for s in specs if s and s.lower() != "env"]


def describe_spec(spec: str) -> str:
    """A provider named without quoting its arguments.

    A secrets command is not itself a secret, but arguments have carried
    tokens before now, so only the scheme and the program are printed.
    """
    scheme, _, rest = spec.partition(":")
    rest = rest.strip()
    name, rest = _split_binding(rest)
    if scheme == "file":
        shown = rest
    else:
        try:
            shown = (shlex.split(rest) or [rest])[0]
        except ValueError:
            shown = rest.split()[0] if rest.split() else rest
    return f"{scheme}: {name + '=' if name else ''}{shown}"


def _split_binding(rest: str) -> tuple[str, str]:
    """`NAME=thing` -> ("NAME", "thing"); anything else -> ("", rest)."""
    head, sep, tail = rest.partition("=")
    if sep and _NAME.match(head.strip()) and tail.strip():
        return head.strip(), tail.strip()
    return "", rest


def parse_payload(text: str) -> dict[str, str]:
    """A secrets manager's output, as names and values.

    JSON object or dotenv lines, because between them that is what every
    manager in the docstring emits. Nested JSON is refused rather than
    flattened by guesswork - a value that is not a string is a payload this
    does not understand, and pretending otherwise stores something wrong.
    """
    text = text.strip()
    if not text:
        return {}
    if text[0] == "{":
        try:
            blob = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SecretsUnavailable(f"output starts with {{ but is not JSON: {exc}") from exc
        if not isinstance(blob, dict):
            raise SecretsUnavailable("JSON output is not an object")
        found = {}
        for name, value in blob.items():
            if isinstance(value, (dict, list)):
                continue  # a structure this does not understand; not guessed at
            found[str(name)] = "" if value is None else str(value)
        return found
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            found[name] = value
    return found


def _read_spec(spec: str) -> dict[str, str]:
    """One provider, fetched. Raises rather than returning nothing quietly."""
    scheme, sep, rest = spec.partition(":")
    scheme, rest = scheme.strip().lower(), rest.strip()
    if not sep or scheme not in ("file", "cmd"):
        raise SecretsUnavailable(
            f"{PROVIDER_VAR} entry {spec!r} is not understood. "
            f"Use file:<path> or cmd:<command>.")
    name, rest = _split_binding(rest)
    if scheme == "file":
        try:
            text = pathlib.Path(rest).expanduser().read_text()
        except OSError as exc:
            raise SecretsUnavailable(f"cannot read {rest}: {exc}") from exc
    else:
        try:
            done = subprocess.run(rest, shell=True, capture_output=True,
                                  text=True, timeout=_timeout())
        except subprocess.TimeoutExpired as exc:
            raise SecretsUnavailable(
                f"{describe_spec(spec)} took longer than {_timeout():g}s") from exc
        except OSError as exc:
            raise SecretsUnavailable(f"{describe_spec(spec)} could not run: {exc}") from exc
        if done.returncode != 0:
            # The provider's own output goes to the LOG and nowhere else.
            #
            # This message ends up in `_STATE["detail"]`, which `report()`
            # returns and `/api/health` serves - and health is deliberately
            # unauthenticated, being the one /api/ path excluded from session
            # handling and the platform's `healthCheckPath`. Two things would
            # have travelled that far: a failing secrets manager's stderr,
            # which routinely names account ids, role ARNs, Vault paths and
            # internal hosts; and, through a stdout fallback that used to be
            # here, the secret itself whenever a wrapper printed the value and
            # then exited non-zero.
            #
            # That contradicted this module's own rule, stated two functions
            # up in `describe_spec` and again in `report`: names and counts
            # only, never a payload. So the classification crosses the HTTP
            # boundary and the diagnostic stays server-side, where an operator
            # reading logs has always been the audience for it.
            detail = (done.stderr or "").strip().splitlines()
            if detail:
                log.error("%s wrote to stderr: %s", describe_spec(spec),
                          detail[-1][:500])
            raise SecretsUnavailable(
                f"{describe_spec(spec)} exited {done.returncode} "
                f"(its output is in the server log, not here)")
        text = done.stdout
    if name:
        value = text.strip()
        return {name: value} if value else {}
    return parse_payload(text)


def load(force: bool = False) -> dict[str, str]:
    """Fetch every configured provider and put the result in the environment.

    Called by `config` before the .env files are read, so the chain in the
    docstring falls out of the order rather than out of a special case: a real
    environment variable is already there and is not touched, the provider
    fills in what is missing, and the files then fill in whatever is left.

    `force` is a refresh: values this module owns are replaced, values the
    operator set by hand are not.
    """
    specs = provider_specs()
    _STATE["configured"] = bool(specs)
    if not specs:
        _STATE.update(state="unset", detail="", names=[])
        return {}
    if not force and _STATE["state"] == "ok" and _ttl() > 0:
        if time.time() - float(_STATE["at"] or 0) < _ttl():
            return {}
    found: dict[str, str] = {}
    origin: dict[str, str] = {}
    for spec in specs:
        try:
            supplied = _read_spec(spec)
        except SecretsUnavailable as exc:
            _STATE.update(state="failed", detail=str(exc), at=time.time(), names=[])
            log.error("%s could not be read: %s", PROVIDER_VAR, exc)
            log.error("  Nothing was taken from it. Whatever the environment and "
                      "the .env files hold is what the app will use.")
            raise
        found.update(supplied)
        # Which provider a value came from, not just that one did. With two
        # entries configured, "the key is stale" and "which of these two am I
        # actually reading" are the same question.
        origin.update({name: describe_spec(spec) for name in supplied})
    applied = []
    for name, value in found.items():
        if not value:
            continue
        # An explicit environment variable outranks the provider, and a refresh
        # must not overwrite one either: someone who exported a key by hand to
        # test something would otherwise have it silently replaced mid-session.
        if name in os.environ and name not in _OWNED:
            continue
        os.environ[name] = value
        _OWNED.add(name)
        SOURCES[name] = f"{PROVIDER_VAR} ({origin[name]})"
        applied.append(name)
    _STATE.update(state="ok", detail=f"{len(applied)} value(s) supplied",
                  at=time.time(), names=sorted(applied))
    if applied:
        log.info("%s supplied %s", PROVIDER_VAR, ", ".join(sorted(applied)))
    _POOL.clear()
    _CURSOR.clear()
    return {name: found[name] for name in applied}


def refresh(reason: str = "") -> bool:
    """Ask the provider again. True if it answered.

    This is what makes "rotate the secret, no redeploy" true rather than
    claimed: the value in force is replaced in the running process.
    """
    if not provider_specs():
        return False
    log.info("re-reading %s%s", PROVIDER_VAR, f" ({reason})" if reason else "")
    try:
        load(force=True)
    except SecretsUnavailable:
        return False
    return True


def pool(name: str) -> list[str]:
    """Every key configured for `name`, in order, without duplicates.

    `ANTHROPIC_API_KEYS` (plural, comma separated) then `ANTHROPIC_API_KEY`.
    One key is a pool of one and behaves exactly as it did before this module
    existed.
    """
    if name in _POOL:
        return _POOL[name]
    keys: list[str] = []
    for raw in (os.environ.get(name + "S", ""), os.environ.get(name, "")):
        for key in (raw or "").split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    _POOL[name] = keys
    _CURSOR.setdefault(name, 0)
    return keys


def active(name: str) -> str:
    """The key in force for `name`, or "" if there is none left.

    Read this rather than `settings.anthropic_api_key` anywhere the value is
    used to make a call: the setting is captured once at import, and a rotation
    or a failover after that would never reach it.
    """
    keys = pool(name)
    index = _CURSOR.get(name, 0)
    return keys[index] if index < len(keys) else ""


def _publish(name: str) -> str:
    """Put the key in force into the environment, and return it.

    This is the seam that makes rotation cost nothing at the call sites. The
    Anthropic SDK resolves `ANTHROPIC_API_KEY` from the environment on every
    client it builds, and `research.py` reads `EXA_API_KEY` there at call time,
    so writing the current key here is enough for both to follow a failover
    without a single one of them knowing this module exists.

    Empty when the pool is exhausted, which is deliberate rather than deleted:
    the SDK treats it as unset and falls back to ANTHROPIC_AUTH_TOKEN or a
    stored `ant auth login` profile, which is the behaviour
    `anthropic_client.build_async_client` documents and keeps.
    """
    key = active(name)
    os.environ[name] = key
    return key


def prime() -> None:
    """Make the environment agree with the configuration.

    `ANTHROPIC_API_KEYS` (plural) is a form only this module reads. The
    Anthropic SDK reads `ANTHROPIC_API_KEY`, so a deployment that configured
    only the pool would have every key listed and none of them sent - the app
    would report demo mode with two perfectly good keys in its environment.

    So the head of each pool is published once, at the end of resolution. It is
    idempotent and it never invents a value: with the singular variable set, or
    with no keys at all, this changes nothing.
    """
    for name in CREDENTIAL_VARS:
        if pool(name) and os.environ.get(name) != active(name):
            _publish(name)


def demote(name: str, reason: str = "") -> str:
    """Step off the key in force and return the next one, or "".

    Called when a key is rejected or capped. With one key this returns "" and
    changes nothing except that the log now says which key stopped working,
    which is the question asked first every time.
    """
    keys = pool(name)
    index = _CURSOR.get(name, 0)
    if index >= len(keys):
        return ""
    _CURSOR[name] = index + 1
    remaining = len(keys) - _CURSOR[name]
    log.warning("%s #%d of %d stopped working%s", name, index + 1, len(keys),
                f": {reason}" if reason else "")
    if remaining:
        log.warning("  failing over to the next key (%d left after it)", remaining - 1)
    else:
        log.error("  no keys left in the pool. Nothing will generate until one is fixed.")
    return _publish(name)


def reset(name: str = "") -> None:
    """Go back to the first key in the pool, and publish it.

    The pool itself is kept rather than recomputed: `demote` writes the key in
    force into the environment, so rebuilding the list from there after a
    failover would rediscover only the key that was failed over *to* and lose
    every one behind it.
    """
    for target in ([name] if name else list(_POOL)):
        _CURSOR[target] = 0
        _publish(target)
    if not name:
        _CURSOR.clear()


def source(name: str) -> str:
    """Where the value in force came from, in words. Never the value itself."""
    if not os.environ.get(name):
        return "nowhere - it is not set"
    if name in SOURCES:
        return SOURCES[name]
    return "the environment"


def report() -> dict:
    """What health, preflight and `setup_key.py --show` all print.

    Names and counts only. A report that prints a secret is a report nobody can
    paste into a bug, which makes it a report nobody runs.
    """
    specs = provider_specs()
    return {
        "provider": [describe_spec(s) for s in specs],
        "configured": bool(specs),
        "state": _STATE["state"],
        "detail": _STATE["detail"],
        "supplied": list(_STATE["names"] or []),
        "ttl_seconds": _ttl(),
        "pools": {name: {"keys": len(pool(name)), "in_use": _CURSOR.get(name, 0) + 1
                         if pool(name) else 0}
                  for name in CREDENTIAL_VARS},
        "sources": {name: source(name) for name in CREDENTIAL_VARS},
    }
