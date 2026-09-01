"""Every class the interface puts in the DOM must have a rule behind it.

Deleting a block of CSS is silent: the markup still renders, the tests still
pass, and the page just looks wrong. That has happened twice on this project -
once taking the wordmark stylesheet, once the whole myFAM block - and neither
time did anything fail. This closes that hole.

Purely structural hooks (a class that exists only for querySelector) are
listed in HOOKS so they do not have to carry a rule they do not need.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "static" / "index.html"

# Classes used as JS selectors only. Anything added here must genuinely need
# no styling of its own - if it looks wrong on the page, it belongs in the CSS.
HOOKS = {
    "echo-icon",    # positioned by .sb-icon; this only names it for toggleEcho
    "typed-offer",  # styled by .mix-topic; marks a topic the listener typed
}


#: A JavaScript expression spliced into a class attribute, e.g.
#: `class="attach-chip' + cls + '"`. The variable name is not a class, and
#: reading it as one asks for a rule that should not exist.
SPLICE = re.compile(r"""(['"])\s*\+.*?\+\s*\1|(['"])\s*\+[^'"]*""", re.S)


def used_classes(html: str) -> set[str]:
    out: set[str] = set()
    for attr in re.findall(r'class="([^"$]*)"', html):
        out.update(SPLICE.sub(" ", attr).split())
    for attr in re.findall(r"class='([^'$]*)'", html):
        out.update(SPLICE.sub(" ", attr).split())
    return {c for c in out if re.fullmatch(r"[A-Za-z][\w-]*", c)}


def stylesheet(html: str) -> str:
    """Every <style> block, with comments stripped.

    Comments matter: a class named only in a `/* ... */` note reads as defined
    and would let a genuinely missing rule through this check unnoticed.
    """
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S))
    return re.sub(r"/\*.*?\*/", " ", css, flags=re.S)


def defined_classes(html: str) -> set[str]:
    return set(re.findall(r"\.([A-Za-z][\w-]*)", stylesheet(html)))


def hidden_needs_guard(html: str) -> list[str]:
    """A rule setting `display` beats the `hidden` attribute silently."""
    css = stylesheet(html)
    shown, guarded = set(), set()
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        display = re.search(r"display\s*:\s*([\w-]+)", body)
        if not display:
            continue
        names = re.findall(r"\.([\w-]+)", selector)
        (guarded if "[hidden]" in selector else shown).update(
            names if display.group(1) != "none" or "[hidden]" in selector else []
        )
    bad = []
    for tag in re.findall(r"<[^>]*>", html):
        if not re.search(r"(?<![-\w])hidden(?=[\s>=])", tag):
            continue
        attr = re.search(r'class="([^"]*)"', tag)
        if not attr:
            continue
        for name in attr.group(1).split():
            if name in shown and name not in guarded:
                bad.append(name)
    return sorted(set(bad))


def main() -> int:
    html = TARGET.read_text(encoding="utf-8")
    missing = sorted(used_classes(html) - defined_classes(html) - HOOKS)
    unguarded = hidden_needs_guard(html)

    if missing:
        print(f"{TARGET.name}: classes used with no CSS rule behind them:")
        for name in missing:
            print(f"  .{name}")
        print("  (a purely structural hook belongs in HOOKS in this file)")
    if unguarded:
        print(f"{TARGET.name}: hidden elements whose class forces them visible:")
        for name in unguarded:
            print(f"  .{name}  - add .{name}[hidden]{{ display:none; }}")
    if missing or unguarded:
        return 1

    print(
        f"checked {len(used_classes(html))} class(es) and every hidden element: OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
