"""Set the API key once, for this machine, and check that it actually works.

    python setup_key.py            # paste it once; stored in ~/.fam/env
    python setup_key.py --show     # where the key came from, and whether it works
    python setup_key.py --remove   # forget it

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

from config import describe_key, key_source, settings, shared_env_path

VAR = "ANTHROPIC_API_KEY"


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


def write_key(key: str, var: str = VAR) -> None:
    """Replace the key line, never append one.

    Two ANTHROPIC_API_KEY lines in one file means "which key is actually being
    sent" depends on who reads it, which has already cost this project a
    session of debugging a perfectly valid key.

    `var` is a parameter so a second provider's key can be stored the same way,
    in the same file, with the same permissions - rather than growing a second
    copy of this function that gets one of those details wrong.
    """
    path = shared_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [
        line for line in read_file()
        if not line.strip().lstrip("export ").startswith(f"{var}=")
    ]
    path.write_text("\n".join(kept + [f"{var}={key}", ""]))
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600: nobody else on this machine
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="report the key in force and test it")
    ap.add_argument("--remove", action="store_true", help="delete the stored key")
    args = ap.parse_args()

    path = shared_env_path()

    if args.remove:
        kept = [line for line in read_file()
                if not line.strip().lstrip("export ").startswith(f"{VAR}=")]
        if path.exists():
            path.write_text("\n".join(kept + [""]) if kept else "")
        print(f"Removed the key from {path}")
        return 0

    if args.show:
        print(f"  stored in : {path}{'' if path.exists() else '  (does not exist yet)'}")
        print(f"  key source: {key_source()}")
        print(f"  key       : {describe_key()}")
        if not settings.anthropic_api_key:
            print("\nNo key. Run: python setup_key.py")
            return 1
        ok, detail = asyncio.run(works(settings.anthropic_api_key))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
