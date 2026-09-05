"""Prove the experiment layer has not touched production FAM.

V1 was approved as additive only. This is the check that keeps it that way:
it diffs the working tree against the base commit and fails if any production
module has been modified. Run it in the loop, the same way `check_css.py`
turned "the page looks wrong" into a failing check.

    python tools/check_additive.py [base-ref]
"""
from __future__ import annotations

import subprocess
import sys

DEFAULT_BASE = "origin/claude/chat-handoff-docs-dl84vm"

#: Everything the app itself runs on. Modifying any of these is a production
#: change and needs explicit approval, not a passing test.
PRODUCTION = {
    "app.py", "pipeline.py", "script_generator.py", "tts.py", "cache.py",
    "config.py", "anthropic_client.py", "audio_utils.py", "topics.py",
    "mixes.py", "social.py", "attachments.py", "voice_store.py",
    "demo_script.py", "write.py",
}
PRODUCTION_DIRS = ("static/",)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    try:
        out = subprocess.run(
            ["git", "diff", "--name-status", base, "--"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        print(f"  SKIPPED: could not diff against {base} ({exc.stderr.strip()[:80]})")
        print("  Pass a base ref explicitly: python tools/check_additive.py <ref>")
        return 0

    changed, modified = [], []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        path = path.strip()
        changed.append(path)
        # Only M and D are production changes. A *new* file next to production
        # code - static/bench.html, say - adds a surface without altering one.
        if status.strip()[:1] in ("M", "D"):
            modified.append(path)

    touched = [
        path for path in modified
        if path in PRODUCTION or any(path.startswith(d) for d in PRODUCTION_DIRS)
    ]

    if touched:
        print(f"  Production files modified against {base}:")
        for path in touched:
            print(f"    {path}")
        print("\n  V1 of the experiment layer is additive only. If one of these")
        print("  changes is intended, it needs approval before it ships.")
        return 1

    # Untracked files are part of the layer too, and are invisible to `git diff`.
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    ).stdout.split()
    layer = [
        path for path in set(changed) | set(untracked)
        if path.startswith("experiments/") or path == "tools/experiment.py"
    ]
    print(f"  additive: {len(set(changed) | set(untracked))} file(s) changed or added, "
          f"none of them a production module")
    print(f"  experiment layer: {len(layer)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
