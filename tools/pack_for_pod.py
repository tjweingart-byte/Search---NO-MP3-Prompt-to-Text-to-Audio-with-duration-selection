#!/usr/bin/env python3
"""Everything the GPU pod needs, in one file, with no credentials in it.

A rented card is somewhere to run FAM for an hour, not somewhere to leave a
GitHub token, an SSH key or an API key. So nothing is cloned on the pod and
nothing is pushed from it: one tarball goes up, one results directory comes
back.

    python tools/pack_for_pod.py
    scp fam-pod.tar.gz root@<pod>:/workspace/

The bundle is `git archive` of the tracked tree at HEAD, plus two files that
are deliberately not in the repository:

  * the reference recording Chatterbox clones, and
  * a **minimal** rights record for it.

The record shipped is regenerated here, not copied. The original names the
person who recorded it; the pod needs only the three answers the engine gates
on. Identity does not leave this machine.

The rights gate runs *before* packing and refuses a recording whose record
does not clear consent, commercial use and synthetic voice - because a gate the
packer can step around is not a gate.

Then on the pod:

    cd /workspace && tar xzf fam-pod.tar.gz && cd FAM
    cat POD.txt
    bash tools/pod_production_test.sh
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
OUT = ROOT / "fam-pod.tar.gz"

#: Where the reference lands inside the bundle. The pod exports
#: CHATTERBOX_REFERENCE at this path; nothing is written outside the extract,
#: so tearing the pod down removes the recording with it.
POD_VOICE_DIR = "FAM/pod-voice"

#: The three answers `ChatterboxEngine.rights_cleared` gates on, and the only
#: fields carried across. Everything else in the source record - who recorded
#: it, when, under what agreement - stays here.
RIGHTS_FIELDS = ("consent", "commercial_use", "synthetic_voice_cleared")

#: Keys that carry identity. If a source record grows one of these, the
#: minimal record must still not carry it, so they are named rather than
#: filtered by guesswork.
IDENTITY_FIELDS = ("name", "speaker", "person", "email", "contact", "source",
                   "recorded_by", "agreement", "notes")


def pod_txt_field(text: str, name: str) -> str:
    """Read one field out of POD.txt.

    POD.txt is the pod's only record of what it is running, because the bundle
    carries no `.git` and `git rev-parse` there exits 128. So the revision and
    the voice digest have to be readable from this text, by a machine rather
    than by eye - `tools/pod_production_test.sh` uses it to verify the packed
    recording is the one that was packed, which was a human comparison before
    and therefore not a check at all.

    Fields are `name`, whitespace, value, at the start of a line. Continuation
    lines are indented and are deliberately not matched.
    """
    for line in text.splitlines():
        if line.startswith(name) and line[len(name):len(name) + 1].isspace():
            return line[len(name):].strip()
    return ""


def voice_digest(text: str) -> str:
    """The packed recording's sha256 prefix, from POD.txt's `voice` line."""
    field = pod_txt_field(text, "voice")
    marker = "sha256 "
    return field.split(marker, 1)[1].strip() if marker in field else ""


def reference_default() -> pathlib.Path:
    from tts import ChatterboxEngine

    return ChatterboxEngine.reference_path()


def rights_for(reference: pathlib.Path) -> pathlib.Path:
    """The rights record for a reference, wherever it was written.

    Production looks for `<reference>.rights.json` beside the recording. Older
    tooling wrote it one directory up, alongside the originals rather than in
    a working folder, so both are tried before giving up.
    """
    for folder in (reference.parent, reference.parent.parent):
        candidate = folder / f"{reference.stem}.rights.json"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"no rights record for {reference.name} in {reference.parent} or "
        f"{reference.parent.parent}.\n"
        "  A cloned voice is somebody's voice: no record, nothing ships.")


def minimal_rights(record: dict, reference: pathlib.Path) -> dict:
    """The record the pod gets: the three answers, and nothing that identifies."""
    out = {field: record[field] for field in RIGHTS_FIELDS}
    out["reference"] = reference.name
    out["note"] = ("Minimal record written by tools/pack_for_pod.py. The full "
                   "record, which identifies the speaker, stays on the "
                   "packing machine and is never copied to rented hardware.")
    return out


def check_reference(reference: pathlib.Path) -> dict:
    """Refuse here, on the machine that has the evidence, not on the pod."""
    if not reference.exists():
        raise SystemExit(
            f"no reference recording at {reference}\n"
            "  Set CHATTERBOX_REFERENCE, or pass --reference.")
    rights = rights_for(reference)
    try:
        record = json.loads(rights.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{rights} is not valid JSON: {exc}") from exc

    for field in RIGHTS_FIELDS:
        value = record.get(field)
        if str(value).strip().lower() not in ("yes", "true"):
            raise SystemExit(
                f"{rights.name} does not clear {field!r} (it says {value!r}).\n"
                "  Refusing to pack the recording. Fix the record, not this check.")

    minimal = minimal_rights(record, reference)
    leaked = [key for key in IDENTITY_FIELDS if key in minimal]
    if leaked:  # pragma: no cover - the allow-list makes this unreachable
        raise SystemExit(f"minimal record would carry identity fields: {leaked}")

    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    size = reference.stat().st_size
    # Bytes below a kilobyte, not "0 KB". A test fixture once printed
    # `reference_3.wav 0 KB` beside a sha nobody recognised, and that line was
    # read as the production voice having been overwritten. A size that rounds
    # away must not look like a recording.
    shown = f"{size} bytes" if size < 1024 else f"{size / 1024:.0f} KB"
    print(f"voice      {reference.name}  {shown}")
    print(f"  sha256   {digest[:16]}  <- must match on the pod")
    print(f"  rights   {rights.name}: consent, commercial use and synthetic "
          "voice all cleared")
    print("           the full record stays here; the pod gets three booleans")
    return {"path": reference, "sha256": digest, "minimal": minimal}


def require_git_checkout() -> None:
    """This tool only works where the history is, which is not the pod.

    The bundle deliberately contains no `.git` - a rented card is somewhere to
    run FAM for an hour, not somewhere to leave a token - so running the packer
    from inside an extracted bundle cannot work, and used to fail as a bare
    `CalledProcessError` from `git rev-parse` with exit 128 and nothing saying
    why. Packing is a Mac-side step; the pod reads the revision out of POD.txt.
    """
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise SystemExit(
            f"{ROOT} is not a git checkout, so there is no HEAD to pack.\n"
            "  If this is an extracted pod bundle: that is by design - it "
            "carries no .git, and the\n"
            "  revision it was built from is the first line of POD.txt. Pack "
            "on the machine that has\n"
            "  the repository, not on the pod.")


def working_tree_is_clean() -> tuple[bool, str]:
    """`git archive` ships HEAD. Uncommitted work would be silently left out."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True)
    return not result.stdout.strip(), result.stdout.strip()


def classify(porcelain: str) -> tuple[list[str], list[str]]:
    """Split `git status --porcelain` into changed-and-tracked, and untracked.

    They are different hazards and deserve different sentences. A modified
    tracked file is the dangerous one: it exists at HEAD in an older form, so
    the bundle would carry a version of a file you are looking at a different
    version of, and nothing about the run would look wrong. An untracked file
    is usually residue - another branch's working files, an editor's leavings -
    but it can also be new source nobody has added yet, which is why it is
    listed rather than passed over.
    """
    changed, untracked = [], []
    for line in porcelain.splitlines():
        (untracked if line.startswith("??") else changed).append(line)
    return changed, untracked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference", default="",
        help="the recording Chatterbox clones. Defaults to what this machine's "
             "settings resolve to, so the pod runs the voice this machine runs.")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="pack HEAD even though the working tree has changes. They will "
             "NOT be in the bundle - git archive ships commits, not files.")
    args = parser.parse_args(argv)

    require_git_checkout()
    clean, dirty = working_tree_is_clean()
    if not clean:
        changed, untracked = classify(dirty)
        if changed:
            print("modified, and tracked at HEAD:")
            print("\n".join("  " + line for line in changed))
        if untracked:
            print("untracked:")
            print("\n".join("  " + line for line in untracked))
        if not args.allow_dirty:
            raise SystemExit(
                "\nRefusing to pack. `git archive` ships HEAD, so nothing "
                "above reaches the pod and the run\n"
                "  would measure code you are not looking at.\n\n"
                "  Commit what belongs on this branch. For anything that does "
                "not belong on it, add an\n"
                "  ignore rule saying why rather than deleting it - another "
                "branch's working files can be\n"
                "  irreplaceable, and this check cannot tell those from "
                "residue.\n\n"
                "  --allow-dirty ships HEAD without the above. It is not a way "
                "past this message; it is\n"
                "  for when you mean to ship an older tree than the one you "
                "have.")
        print("  --allow-dirty: shipping HEAD anyway, without the above\n")

    reference = (pathlib.Path(args.reference).expanduser() if args.reference
                 else reference_default())
    voice = check_reference(reference)

    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        tracked = tmpdir / "tracked.tar"
        with tracked.open("wb") as handle:
            # The tracked tree at HEAD: no .git, no .env, no credentials.
            subprocess.run(
                ["git", "-C", str(ROOT), "archive", "--format=tar",
                 "--prefix=FAM/", "HEAD"],
                stdout=handle, check=True)

        rights_file = tmpdir / f"{reference.stem}.rights.json"
        rights_file.write_text(json.dumps(voice["minimal"], indent=2) + "\n",
                               encoding="utf-8")
        note = tmpdir / "POD.txt"
        note.write_text(
            f"revision  {revision}\n"
            f"branch    {branch}\n"
            f"voice     {reference.name}, sha256 {voice['sha256'][:16]}\n"
            f"          rights cleared at pack time; the full record, which\n"
            f"          identifies the speaker, stayed on the packing machine.\n"
            "\n"
            "No git remote, GitHub token or SSH key is included or needed.\n"
            "ANTHROPIC_API_KEY is exported into the pod shell by hand and is\n"
            "never written into this bundle.\n"
            "\n"
            "Next:  bash tools/pod_production_test.sh\n",
            encoding="utf-8")

        with tarfile.open(OUT, "w:gz") as bundle:
            with tarfile.open(tracked, "r") as inner:
                for member in inner.getmembers():
                    bundle.addfile(member, inner.extractfile(member)
                                   if member.isfile() else None)
            bundle.add(voice["path"],
                       arcname=f"{POD_VOICE_DIR}/{reference.name}")
            bundle.add(rights_file,
                       arcname=f"{POD_VOICE_DIR}/{rights_file.name}")
            bundle.add(note, arcname="FAM/POD.txt")

    size = OUT.stat().st_size / 1024 ** 2
    print(f"\nwrote      {OUT.name}  ({size:.1f} MB)")
    print(f"  revision {revision[:12]} on {branch}")
    print("\nnext")
    print(f"  scp {OUT.name} root@<pod>:/workspace/")
    print("  ssh root@<pod>")
    print("  cd /workspace && tar xzf fam-pod.tar.gz && cd FAM && cat POD.txt")
    print("  bash tools/pod_production_test.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
