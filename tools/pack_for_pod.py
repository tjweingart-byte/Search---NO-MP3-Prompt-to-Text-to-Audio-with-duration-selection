#!/usr/bin/env python3
"""Everything the GPU pod needs, in one file, with no credentials.

A rented card is somewhere to run a benchmark, not somewhere to leave a
GitHub token. This packs the tracked tree plus the one git-ignored file the
benchmark needs - the validated chunk corpus - into a single tarball you copy
across. The pod needs no git remote, no token and no SSH key.

It validates the corpus *before* packing, because shipping the wrong one would
silently break the comparison with the Mac run and nobody would notice until
the numbers were already paid for.

    python tools/pack_for_pod.py
    python tools/pack_for_pod.py --reference experiments/references/working/reference_1.wav
    scp fam-pod.tar.gz root@<pod>:/workspace/

Then on the pod:

    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "experiments" / "chunks" / "first_chunks.json"
OUT = ROOT / "fam-pod.tar.gz"

#: The pod must run the same corpus the Mac ran, or the comparison is void.
EXPECTED_MIN_CHUNKS = 1


def check_corpus() -> dict:
    if not CORPUS.exists():
        raise SystemExit(
            f"no corpus at {CORPUS}\n"
            "  python tools/extract_chunks.py experiments/results/warm_first_token")
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    if len(chunks) < EXPECTED_MIN_CHUNKS:
        raise SystemExit(f"corpus holds {len(chunks)} chunks; nothing to pack")

    cut = [c for c in chunks if not c.get("ends_complete", True)]
    if cut:
        raise SystemExit(
            f"{len(cut)} chunk(s) in the corpus do not end at a sentence "
            "boundary. Refusing to ship a corpus the Mac run would not have "
            "used. Re-extract first.")

    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()[:16]
    buckets = {}
    for chunk in chunks:
        buckets.setdefault(chunk["bucket"], []).append(chunk["words"])
    print(f"corpus     {len(chunks)} chunks, CUT included 0")
    for name in ("short", "medium", "long"):
        mine = buckets.get(name)
        if mine:
            print(f"  {name:<8} {len(mine):>3}  ({min(mine)}-{max(mine)} words)")
    print(f"  excluded {len(data.get('excluded') or [])}")
    print(f"  sha256   {digest}  <- must match on the pod")
    return {"chunks": len(chunks), "sha256": digest}


def find_rights(path: pathlib.Path) -> pathlib.Path:
    """The rights record for a reference, wherever the checker put it.

    `check_reference_audio.py` writes `<references>/reference_N.rights.json`,
    beside the recordings rather than inside `working/`, so both places are
    tried and the search stops at the references root.
    """
    for folder in (path.parent, path.parent.parent):
        candidate = folder / f"{path.stem}.rights.json"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"no rights record for {path.name} in {path.parent} or "
        f"{path.parent.parent}. Run tools/check_reference_audio.py and populate "
        "it before shipping the recording anywhere.")


def check_reference(path: pathlib.Path) -> dict:
    """The one reference recording the run needs, and nothing else from there.

    `experiments/references/` also holds `sources.json`, which maps neutral ids
    back to the people who recorded them, and the rights records, which name
    them outright. Neither belongs on a rented machine, so exactly one audio
    file is added, by name, and the rest of the directory is never walked. The
    rights record is *read* here and left behind; the pod gets the neutral
    filename and the fact that the gate passed.

    A recording whose record does not clear consent, commercial use and
    synthetic voice is refused here rather than on the pod, because a gate the
    packer can step around is not a gate.
    """
    if not path.exists():
        raise SystemExit(f"no reference recording at {path}")
    rights = find_rights(path)
    record = json.loads(rights.read_text(encoding="utf-8"))
    for field in ("consent", "commercial_use", "synthetic_voice_cleared"):
        value = record.get(field)
        if str(value).strip().lower() not in ("yes", "true"):
            raise SystemExit(
                f"{rights.name} does not clear {field!r} (it says {value!r}). "
                "Refusing to pack the recording. Fix the record, not this check.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"reference  {path.name}  {path.stat().st_size / 1024:.0f} KB  "
          f"sha256 {digest}")
    print(f"  rights   {rights.name}: consent, commercial use and synthetic "
          "voice all cleared (record stays here)")
    return {"name": path.name, "sha256": digest, "path": path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", default="",
                        help="also pack this one reference recording, for a run "
                             "that clones a voice. Its rights record must clear "
                             "consent, commercial use and synthetic voice.")
    args = parser.parse_args()

    summary = check_corpus()
    reference = check_reference(pathlib.Path(args.reference)) if args.reference else None

    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        tracked = pathlib.Path(tmp) / "tracked.tar"
        with tracked.open("wb") as handle:
            # The tracked tree at HEAD, with no .git and no credentials in it.
            subprocess.run(
                ["git", "-C", str(ROOT), "archive", "--format=tar",
                 "--prefix=FAM/", "HEAD"],
                stdout=handle, check=True)

        with tarfile.open(OUT, "w:gz") as bundle:
            with tarfile.open(tracked, "r") as inner:
                for member in inner.getmembers():
                    bundle.addfile(member, inner.extractfile(member)
                                   if member.isfile() else None)
            # The corpus is git-ignored, so it is added by hand.
            bundle.add(CORPUS, arcname="FAM/experiments/chunks/first_chunks.json")
            reference_line = ""
            if reference:
                bundle.add(reference["path"],
                           arcname="FAM/experiments/references/working/"
                                   + reference["name"])
                reference_line = (
                    f"voice    {reference['name']}, sha256 "
                    f"{reference['sha256']}\n"
                    "         rights cleared at pack time; the record itself,\n"
                    "         and sources.json, stay off the pod.\n")
            note = pathlib.Path(tmp) / "POD.txt"
            note.write_text(
                f"revision {revision}\n"
                f"corpus   {summary['chunks']} chunks, sha256 "
                f"{summary['sha256']}\n"
                f"{reference_line}"
                "No git remote and no GitHub token is included or needed.\n"
                "API keys, where a run needs them, are exported into the pod\n"
                "shell by hand and are never written into this bundle.\n",
                encoding="utf-8")
            bundle.add(note, arcname="FAM/POD.txt")

    size = OUT.stat().st_size / 1024 ** 2
    print(f"\nwrote      {OUT.name}  ({size:.1f} MB)")
    print(f"  revision {revision[:12]}")
    print("\nnext")
    print(f"  scp {OUT.name} root@<pod>:/workspace/")
    print("  ssh root@<pod>")
    print("  cd /workspace && tar xzf fam-pod.tar.gz && cd FAM")
    print("  python tools/extract_chunks.py --verify    # same count, same buckets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
