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


async def works(key: str) -> tuple[bool, str, str]:
    """Speak one short line with it. Returns (accepted, detail, local_warning).

    Two different questions, kept apart on purpose:

    * **Did WellSaid accept this key?** That decides whether it is stored.
    * **Can this machine play what came back?** That does not.

    Running them together is what made a perfectly good key be refused: if
    WellSaid served MP3 and ffmpeg was not installed, the decode failed, the
    whole check reported REJECTED, and nothing was stored - so the fix
    ("install ffmpeg") looked like a key problem and the key had to be pasted
    all over again afterwards.

    This is the project's own "verify, do not inspect" rule overshooting in
    the other direction. A check must perform the real action, but it must
    answer the question it was asked and no more.
    """
    os.environ[VAR] = key
    # settings is frozen and was built at import time, so rebuild it to pick
    # up the key that was just typed.
    import config

    config.settings = config.Settings()
    import importlib

    import wellsaid

    importlib.reload(wellsaid)

    speaker = config.settings.wellsaid_chase_j_id
    engine = wellsaid.WellSaidEngine()
    text, chunk = "Testing the FAM voice.", None
    try:
        # The request on its own, so the answer to "is the key good" does not
        # depend on anything installed locally.
        chunk = await engine._speak_one(text, speaker, 1, 1)
    except wellsaid.WellSaidUnreachable as exc:
        # A network that cannot get to their API says nothing at all about the
        # key. Calling this "rejected" sends someone off to regenerate a key
        # that was fine.
        return False, f"UNREACHABLE::{str(exc)[:300]}", ""
    except wellsaid.WellSaidLocalError as exc:
        # WellSaid answered - the key is good. Only this machine could not use
        # the answer, which is a separate thing to fix and not a reason to
        # throw the key away and make it be pasted again afterwards.
        return True, f"the key is accepted by WellSaid (speaker {speaker} answered).", str(exc)
    except wellsaid.WellSaidError as exc:
        return False, f"{str(exc)[:300]}", ""
    except Exception as exc:  # noqa: BLE001 - the reason is the whole point
        return False, f"{type(exc).__name__}: {str(exc)[:300]}", ""

    if not chunk:
        return False, "WellSaid accepted the request but returned no audio.", ""
    seconds = len(chunk) / float(config.settings.sample_rate * 2)
    return True, f"spoke {seconds:.1f}s of audio as speaker {speaker}.", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="report the key and test it")
    ap.add_argument("--remove", action="store_true", help="delete the stored key")
    ap.add_argument("--no-verify", action="store_true",
                    help="store the key without checking it first (for a network "
                         "that blocks api.wellsaidlabs.com)")
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
        ok, detail, warning = asyncio.run(works(settings.wellsaid_api_key))
        print(f"  accepted  : {'YES - ' + detail if ok else 'NO - ' + detail}")
        if warning:
            print(f"  but note  : {warning}")
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

    if args.no_verify:
        write_key(key, VAR)
        print(f"\nStored in {path} WITHOUT checking it - you asked for --no-verify.")
        print("If it is wrong, the first WellSaid episode will fail with the reason.")
        return 0

    print("\nAsking WellSaid to speak one line with it, before storing it…")
    ok, detail, warning = asyncio.run(works(key))
    if not ok and detail.startswith("UNREACHABLE::"):
        print(f"\nCOULD NOT REACH WELLSAID - {detail.split('::', 1)[1]}")
        print(f"  key tried: {describe(key)}")
        print("\nThis says nothing about your key - the request never arrived.")
        print("  Check your internet connection, or whether a VPN, firewall or")
        print("  work network is blocking api.wellsaidlabs.com.")
        print("\nIf you are sure the key is right and want to store it without")
        print("  checking it first:  python setup_wellsaid.py --no-verify")
        return 1
    if not ok:
        print(f"\nREJECTED BY WELLSAID - {detail}")
        print(f"  key tried: {describe(key)}")
        print("\nNothing was stored. A key WellSaid will not accept is worse stored")
        print("  than not: the voices would appear in the picker and fail on the tap.")
        return 1

    write_key(key, VAR)
    print(f"\nAccepted - {detail}")
    print(f"Stored in {path} (readable only by you).")
    if warning:
        # Stored anyway. The key is good; this machine just cannot decode what
        # WellSaid sends yet, and that is a separate thing to fix once.
        print("\nONE MORE STEP before you can hear it:")
        print(f"  {warning}")
        print("\nYour key is saved, so you will not have to paste it again -")
        print("  fix the above and run ./start.sh.")
    else:
        print("\nStart the app with ./start.sh, then pick 'Chase J (WellSaid)' or")
        print("'Kai M (WellSaid)' from the Voice list in the player.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
