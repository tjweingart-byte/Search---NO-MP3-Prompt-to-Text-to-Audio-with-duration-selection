"""Store the WellSaid API key once, for this machine, and prove it works.

    python setup_wellsaid.py            # paste it once; stored in ~/.fam/env
    python setup_wellsaid.py --show     # is a key set, and does it work?
    python setup_wellsaid.py --remove   # forget it

Same place and same rules as the Anthropic key: `~/.fam/env`, chmod 600,
outside the project folder so it is never committed and survives a new copy of
the app. It is never written into any source file.

Nothing is stored until WellSaid actually speaks a word with it. "A key is
set" is not "the key works", and this project has already paid for that
distinction more than once - so this renders one short line of real audio with
the real speaker id and only writes the key if audio comes back.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import os

from config import settings, shared_env_path
from setup_key import read_file, write_key

VAR = "WELLSAID_API_KEY"


def describe(key: str = "") -> str:
    """A safe fingerprint. Never enough to be a secret."""
    key = key or os.environ.get(VAR, "")
    if not key:
        return "no key configured"
    return f"{key[:4]}...{key[-4:]} ({len(key)} chars)"


async def works(key: str) -> tuple[bool, str]:
    """Speak one short line with it, and report exactly what came back."""
    os.environ[VAR] = key
    # settings is frozen and was built at import time, so rebuild it to pick
    # up the key that was just typed.
    import config

    config.settings = config.Settings()
    import importlib

    import wellsaid

    importlib.reload(wellsaid)

    speaker = config.settings.wellsaid_chase_j_id
    try:
        pcm = await wellsaid.WellSaidEngine().synth(
            "Testing the FAM voice.", 150, f"wellsaid:{speaker}")
    except Exception as exc:  # noqa: BLE001 - the reason is the whole point
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    if not pcm:
        return False, "WellSaid accepted the request but returned no audio."
    seconds = len(pcm) / float(config.settings.sample_rate * 2)
    return True, f"spoke {seconds:.1f}s of audio as speaker {speaker}."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="report the key and test it")
    ap.add_argument("--remove", action="store_true", help="delete the stored key")
    args = ap.parse_args()

    path = shared_env_path()

    if args.remove:
        kept = [line for line in read_file()
                if not line.strip().lstrip("export ").startswith(f"{VAR}=")]
        if path.exists():
            path.write_text("\n".join(kept + [""]) if kept else "")
        print(f"Removed the WellSaid key from {path}")
        return 0

    if args.show:
        print(f"  stored in : {path}{'' if path.exists() else '  (does not exist yet)'}")
        print(f"  key       : {describe(settings.wellsaid_api_key)}")
        print(f"  endpoint  : {settings.wellsaid_api_url}")
        print(f"  Chase J   : speaker_id {settings.wellsaid_chase_j_id}")
        print(f"  Kai M     : speaker_id {settings.wellsaid_kai_m_id}")
        if not settings.wellsaid_api_key:
            print("\nNo key. Run: python setup_wellsaid.py")
            return 1
        ok, detail = asyncio.run(works(settings.wellsaid_api_key))
        print(f"  accepted  : {'YES - ' + detail if ok else 'NO - ' + detail}")
        return 0 if ok else 1

    print(f"The WellSaid key is stored once, in {path}, and every copy of the app")
    print("reads it. It is never written into the source, and never committed.\n")
    print("Get one from your WellSaid Labs account: Settings -> API (studio.wellsaidlabs.com).\n")
    if settings.wellsaid_api_key:
        print(f"There is already a key ({describe(settings.wellsaid_api_key)}).")
        print("Entering a new one replaces it.\n")

    try:
        key = getpass.getpass("Paste your WellSaid API key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNothing changed.")
        return 1
    if not key:
        print("Nothing entered; nothing changed.")
        return 1

    print("\nAsking WellSaid to speak one line with it, before storing it…")
    ok, detail = asyncio.run(works(key))
    if not ok:
        print(f"\nREJECTED - {detail}")
        print(f"  key tried: {describe(key)}")
        print("\nNothing was stored. A key that does not work is worse stored than not:")
        print("  the voices would appear in the picker and fail when you tapped one.")
        return 1

    write_key(key, VAR)
    print(f"\nAccepted - {detail}")
    print(f"Stored in {path} (readable only by you).")
    print("\nStart the app with ./run.sh, then pick 'Chase J (WellSaid)' or")
    print("'Kai M (WellSaid)' from the Voice list in the player.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
