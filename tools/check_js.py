"""Parse every inline <script> in the interface.

pytest never opens static/index.html, so a stray brace there passes CI and
breaks the app. Uses node when it is available and falls back to a bracket
balance check when it is not, so this is never a reason a machine cannot run
the checks.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "static" / "index.html"]
PLAIN_JS = [ROOT / "static" / "fam-audio.js"]

INLINE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def sources() -> list[tuple[str, str]]:
    out = [(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")) for p in PLAIN_JS]
    for path in TARGETS:
        html = path.read_text(encoding="utf-8")
        blocks = INLINE.findall(html)
        if not blocks:
            raise SystemExit(f"{path.name}: no inline script found - has the file changed?")
        out.append((f"{path.name} (inline)", "\n".join(blocks)))
    return out


def main() -> int:
    node = shutil.which("node")
    failures = []
    for name, code in sources():
        if node:
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
                fh.write(code)
                tmp = fh.name
            result = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
            pathlib.Path(tmp).unlink(missing_ok=True)
            if result.returncode != 0:
                failures.append(f"{name}:\n{result.stderr.strip()}")
        else:
            depth = {"{": 0, "(": 0, "[": 0}
            pairs = {"}": "{", ")": "(", "]": "["}
            for ch in re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S):
                if ch in depth:
                    depth[ch] += 1
                elif ch in pairs:
                    depth[pairs[ch]] -= 1
            bad = [k for k, v in depth.items() if v != 0]
            if bad:
                failures.append(f"{name}: unbalanced {', '.join(bad)} (node not installed, "
                                "so this is only a balance check)")
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"checked {len(sources())} script source(s): OK"
          + ("" if node else "  [balance check only - node not installed]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
