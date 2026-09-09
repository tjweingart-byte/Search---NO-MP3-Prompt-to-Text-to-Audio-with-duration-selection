"""Set an API key once, for this machine, and check that it actually works.

    python setup_key.py                  # the Anthropic key, which writes episodes
    python setup_key.py --exa            # the Exa key, which researches them
    python setup_key.py --show           # where the key came from, and whether it works
    python setup_key.py --show --exa
    python setup_key.py --remove [--exa] # forget it

Two credentials, one file, one habit. `--exa` is needed when
RESEARCH_BACKEND=exa - which is the default - and a researched episode fails
without it rather than quietly searching another way.

The key goes in `~/.fam/env`, next to the shared voice store and for the same
reason: it lives **outside the project folder**, so unpacking a new copy of the
app finds it already there. A key kept in a project `.env` is lost every time
the app moves, and the workaround for that is pasting the key again - into a
terminal, into a chat window, into whatever is to hand. Once per machine, and
never again.

It is deliberately **not** written into any source file. Source gets committed,
and a key in a commit is a key that has to be rotated - it stays in the history
even after the line is deleted. `~/.fam/env` is chmod 600 and is not in, near,
or reachable from the repository.

Nothing is stored until Claude confirms the key is accepted, so "no key
detected" and "bad key" both get answered here rather than at play time.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import stat
import sys

import credentials
from config import describe_key, key_source, settings, shared_env_path

VAR = "ANTHROPIC_API_KEY"
EXA_VAR = "EXA_API_KEY"


async def exa_works(key: str) -> tuple[bool, str]:
    """Ask Exa whether it accepts this key, before writing it anywhere.

    A one-result search, which is the cheapest call that proves the credential
    - about half a cent. Exa has no free "who am I" endpoint, so checking costs
    something; a key stored without checking costs more. "A key is set" is not
    "the key works" (PROBLEMS.md 52), and this is the one place where finding
    out is cheap.
    """
    os.environ[EXA_VAR] = key
    try:
        from exa_py import Exa
    except ImportError:
        return False, ("exa_py is not installed, so the key cannot be checked. "
                       "`pip install -r requirements-exa.txt`")
    try:
        reply = await asyncio.to_thread(
            lambda: Exa(key).search_and_contents(
                "test", type="fast", num_results=1, highlights=True))
    except Exception as exc:  # noqa: BLE001 - the reason matters, not the class
        detail = getattr(exc, "message", "") or str(exc)
        return False, f"{type(exc).__name__}: {detail[:200]}"
    found = len(list(getattr(reply, "results", []) or []))
    return True, f"Exa accepted it and returned {found} result(s)."


def describe_exa_key(key: str = "") -> str:
    """A safe fingerprint, for the same reason `describe_key` exists: an error
    about the wrong key looks identical whichever wrong key produced it."""
    key = key or os.environ.get(EXA_VAR, "")
    if not key:
        return "not set"
    return f"{len(key)} chars, ending {key[-4:]}"


async def works(key: str) -> tuple[bool, str]:
    """Ask Claude whether it accepts this key, before writing it anywhere.

    `models.retrieve` bills nothing and answers the two questions that matter:
    is this key accepted, and can this account use the model the app is set to.
    """
    os.environ[VAR] = key
    try:
        import anthropic  # noqa: F401

        from anthropic_client import build_async_client

        await build_async_client(key).models.retrieve(settings.model)
    except Exception as exc:  # noqa: BLE001 - the reason matters, not the class
        detail = getattr(exc, "message", "") or str(exc)
        return False, f"{type(exc).__name__}: {detail[:200]}"
    return True, f"{settings.model} is reachable with this key."


def read_file() -> list[str]:
    try:
        return shared_env_path().read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def without(name: str) -> list:
    """Every line of the shared file except the one setting `name`."""
    return [line for line in read_file()
            if not line.strip().lstrip("export ").startswith(f"{name}=")]


def write_key(key: str, name: str = VAR) -> None:
    """Replace the key line, never append one.

    Two lines setting the same variable means "which key is actually being
    sent" depends on who reads it, which has already cost this project a
    session of debugging a perfectly valid key. The other variable's line is
    preserved untouched - storing one credential must never drop the other.
    """
    path = shared_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = without(name)
    path.write_text("\n".join(kept + [f"{name}={key}", ""]))
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600: nobody else on this machine
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="report the key in force and test it")
    ap.add_argument("--remove", action="store_true", help="delete the stored key")
    ap.add_argument("--exa", action="store_true",
                    help="the Exa key, which researches episodes, rather than "
                         "the Anthropic one, which writes them")
    args = ap.parse_args()

    path = shared_env_path()
    name = EXA_VAR if args.exa else VAR

    if args.remove:
        kept = without(name)
        if path.exists():
            path.write_text("\n".join(kept + [""]) if kept else "")
        print(f"Removed {name} from {path}")
        if args.exa:
            print("Researched episodes will now fail unless you also set "
                  "RESEARCH_BACKEND=claude.")
        return 0

    if args.exa:
        return exa_main(path, args.show)

    if args.show:
        print(f"  stored in : {path}{'' if path.exists() else '  (does not exist yet)'}")
        print(f"  key source: {key_source()}")
        print(f"  key       : {describe_key()}")
        report = credentials.report()
        if report["configured"]:
            print(f"  provider  : {', '.join(report['provider'])} - {report['state']}"
                  f"{': ' + report['detail'] if report['detail'] else ''}")
        pooled = report["pools"]["ANTHROPIC_API_KEY"]["keys"]
        if pooled > 1:
            print(f"  pool      : {pooled} keys, #{report['pools']['ANTHROPIC_API_KEY']['in_use']} "
                  f"in use. Failover only - Anthropic rate limits are per")
            print("              organisation, so extra keys buy no extra headroom.")
        key = credentials.active(VAR) or settings.anthropic_api_key
        if not key:
            print("\nNo key. Run: python setup_key.py")
            print(f"To stop every new machine asking, set {credentials.PROVIDER_VAR} "
                  f"instead - see CREDENTIALS.md.")
            return 1
        ok, detail = asyncio.run(works(key))
        print(f"  accepted  : {'YES - ' + detail if ok else 'NO - ' + detail}")
        return 0 if ok else 1

    print(f"The key is stored once, in {path}, and every copy of the app reads it.")
    print("It is never written into the source, and never committed.\n")
    if settings.anthropic_api_key:
        print(f"There is already a key ({describe_key()}) from {key_source()}.")
        print("Entering a new one replaces it.\n")

    try:
        # getpass, so the key is not echoed into the terminal - and therefore
        # not into a screenshot or scrollback shared with anyone later.
        key = getpass.getpass("Paste your Anthropic API key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNothing changed.")
        return 1
    if not key:
        print("Nothing entered; nothing changed.")
        return 1
    if not key.startswith("sk-ant-"):
        print(f"\nThat does not look like an API key - {describe_key(key)}.")
        print("API keys start with sk-ant- and come from console.anthropic.com.")
        print("Checking it anyway, in case the format has changed.\n")

    print("Checking it with Claude before storing it…")
    ok, detail = asyncio.run(works(key))
    if not ok:
        print(f"\nREJECTED - {detail}")
        print(f"  key tried: {describe_key(key)}")
        print("\nNothing was stored. A key that does not work is worse stored than "
              "not stored:\n  the app would start, say it was live, and fail on the "
              "first episode.")
        return 1

    write_key(key)
    print(f"\nAccepted - {detail}")
    print(f"Stored in {path} (readable only by you).")
    print("\nYou will not be asked again on this machine, including by a new copy "
          "of the app.\nCheck it any time with:  python setup_key.py --show")
    if not credentials.provider_specs():
        print("\nThis machine is done. Every OTHER machine - a pod, a container, a "
              "CI runner,\na colleague's laptop - is a fresh ~/.fam and will ask "
              f"again. One variable\nends that for all of them: set "
              f"{credentials.PROVIDER_VAR}. See CREDENTIALS.md.")
    return 0


def exa_main(path, show: bool) -> int:
    """The Exa half. Deliberately the same shape as the Anthropic half above -
    verify, then store, and store nothing that does not work."""
    stored = os.environ.get(EXA_VAR, "")

    if show:
        print(f"  stored in : {path}{'' if path.exists() else '  (does not exist yet)'}")
        print(f"  backend   : RESEARCH_BACKEND={settings.research_backend}")
        print(f"  key       : {describe_exa_key()}")
        if not stored:
            print("\nNo Exa key. Run: python setup_key.py --exa")
            if settings.research_backend == "exa":
                print("Researched episodes will fail until you do, or until "
                      "you set RESEARCH_BACKEND=claude.")
            return 1
        print("  checking it with Exa (one search, about half a cent)…")
        ok, detail = asyncio.run(exa_works(stored))
        print(f"  accepted  : {'YES - ' + detail if ok else 'NO - ' + detail}")
        return 0 if ok else 1

    print(f"The Exa key is stored once, in {path}, and every copy of the app "
          "reads it.")
    print("It is never written into the source, and never committed.\n")
    print("It is needed because RESEARCH_BACKEND defaults to `exa`: a "
          "researched episode\nretrieves first and Claude reads the packet. "
          "Without a key those episodes fail\nrather than quietly searching "
          "another way.\n")
    if stored:
        print(f"There is already an Exa key ({describe_exa_key()}).")
        print("Entering a new one replaces it.\n")

    try:
        key = getpass.getpass("Paste your Exa API key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNothing changed.")
        return 1
    if not key:
        print("Nothing entered; nothing changed.")
        return 1

    print("Checking it with Exa before storing it (one search, about half a "
          "cent)…")
    ok, detail = asyncio.run(exa_works(key))
    if not ok:
        print(f"\nREJECTED - {detail}")
        print(f"  key tried: {describe_exa_key(key)}")
        print("\nNothing was stored. A key that does not work is worse stored "
              "than not stored:\n  the server would start, report research as "
              "configured, and fail on the first\n  researched episode.")
        return 1

    write_key(key, EXA_VAR)
    print(f"\nAccepted - {detail}")
    print(f"Stored in {path} (readable only by you).")
    print("\nYou will not be asked again on this machine, including by a new "
          "copy of the app.\nCheck it any time with:  python setup_key.py "
          "--show --exa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
