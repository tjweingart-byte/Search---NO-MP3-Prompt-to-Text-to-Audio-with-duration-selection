#!/usr/bin/env python3
"""Write one reference's rights record, matched by the source recording.

Matching by source filename rather than by neutral id is the point. The ids are
assigned by a shuffle, and after equalising `sources.json` points at
`working/reference_N.wav` - so hand-editing the right file means first working
out which one it is, and getting that wrong would attach a person's consent to
somebody else's voice.

Give it the name the recording arrived under and it finds the rest.

    python tools/set_rights.py experiments/references \\
        --source Ian --speaker "Ian Solomon" \\
        --consent yes --commercial-use yes --synthetic-voice-cleared yes \\
        --recorded-for "recorded 2026-09-08 for the FAM synthetic-voice experiment" \\
        --notes "Shipping with FAM permitted. Permission record: written and verbal."

Values are written exactly as given. Nothing is inferred: there is no default
for any of the three clearance flags, and omitting one is an error rather than
a no.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.check_reference_audio import SOURCES, load_sources  # noqa: E402

WORKING = "working"


def source_names(folder: pathlib.Path) -> dict:
    """Neutral id -> the name the recording arrived under.

    After equalising, `sources.json` points at the working copies, so the
    original names live in `working/MANIFEST.json`. Before equalising they are
    in `sources.json` itself. Both are handled, because rights can reasonably
    be filled in at either point.
    """
    manifest = folder / WORKING / "MANIFEST.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {key: entry["source"]
                for key, entry in (data.get("references") or {}).items()}
    return {key: path.name for key, path in load_sources(folder).items()}


def find(folder: pathlib.Path, source: str) -> str:
    """The neutral id whose recording matches `source`, or a refusal."""
    names = source_names(folder)
    wanted = source.strip().lower()
    matches = [key for key, name in names.items()
               if name.lower() == wanted
               or pathlib.Path(name).stem.lower() == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"no recording called {source!r} in {folder}\n"
            "  known: " + ", ".join(sorted(names.values())) + "\n"
            f"  (run check_reference_audio.py {folder} --adopt first if "
            f"{SOURCES} does not exist)")
    raise SystemExit(f"{source!r} matches more than one recording: "
                     + ", ".join(matches))


def yes_no(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("yes", "y", "true"):
        return True
    if lowered in ("no", "n", "false"):
        return False
    raise argparse.ArgumentTypeError(
        f"{value!r} is neither yes nor no. 'not yet' is a no; say no.")


def build_parser() -> argparse.ArgumentParser:
    """Exposed so a test can inspect the real actions rather than the source.

    What matters is that the three clearance flags are required and have no
    default: a permission that was never stated must not become a `true`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="experiments/references")
    parser.add_argument("--source",
                        help="the name the recording arrived under, e.g. Ian")
    parser.add_argument("--speaker")
    parser.add_argument("--speakers",
                        help="all of them at once, as "
                             "'Ian=Ian Solomon,TJ=TJ Weingart'. The clearances "
                             "below then apply to every named recording, so use "
                             "it only when they really are the same.")
    parser.add_argument("--recorded-for", required=True, dest="recorded_for",
                        help="goes into 'source': where the recording came "
                             "from and what it was made for")
    # No defaults. A clearance that was never stated must not become a true.
    parser.add_argument("--consent", required=True, type=yes_no)
    parser.add_argument("--commercial-use", required=True, type=yes_no,
                        dest="commercial_use")
    parser.add_argument("--synthetic-voice-cleared", required=True,
                        type=yes_no, dest="synthetic_voice_cleared")
    parser.add_argument("--notes", required=True)
    return parser


def parse_speakers(text: str) -> dict:
    """'Ian=Ian Solomon,TJ=TJ Weingart' -> {'Ian': 'Ian Solomon', ...}"""
    pairs = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(
                f"{chunk!r} is not 'recording=Speaker Name'. Expected "
                "something like 'Ian=Ian Solomon,TJ=TJ Weingart'.")
        source, speaker = chunk.split("=", 1)
        source, speaker = source.strip(), speaker.strip()
        if not source or not speaker:
            raise SystemExit(f"{chunk!r} has an empty side")
        pairs[source] = speaker
    if not pairs:
        raise SystemExit("--speakers is empty")
    return pairs


def write_record(folder: pathlib.Path, source: str, speaker: str,
                 args) -> tuple:
    key = find(folder, source)
    record = {
        "source": args.recorded_for,
        "speaker": speaker,
        "consent": args.consent,
        "commercial_use": args.commercial_use,
        "synthetic_voice_cleared": args.synthetic_voice_cleared,
        "notes": args.notes,
    }
    (folder / f"{key}.rights.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return key, record


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.speakers) == bool(args.source):
        raise SystemExit("give either --source with --speaker, or --speakers")
    if args.source and not args.speaker:
        raise SystemExit("--source needs --speaker")

    folder = pathlib.Path(args.folder)
    people = (parse_speakers(args.speakers) if args.speakers
              else {args.source: args.speaker})

    # Resolve every recording before writing any file: a typo in the third
    # name should not leave the first two written and the set half-done.
    resolved = {source: find(folder, source) for source in people}
    if len(set(resolved.values())) != len(resolved):
        raise SystemExit(f"two names resolved to the same recording: {resolved}")

    written = []
    for source, speaker in people.items():
        written.append((*write_record(folder, source, speaker, args), source))

    print()
    for key, record, source in written:
        print(f"  {key}  <-  {source:<12}  {record['speaker']}")
        blocked = [f for f in ("consent", "commercial_use",
                               "synthetic_voice_cleared") if not record[f]]
        if blocked:
            print(f"      BLOCKED: {', '.join(blocked)} is false")
    print(f"\n  wrote {len(written)} rights record(s)")
    print(f"  source: {args.recorded_for}")
    print(f"  notes:  {args.notes}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
