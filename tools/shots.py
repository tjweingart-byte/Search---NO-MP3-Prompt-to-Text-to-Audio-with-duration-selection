"""Photograph every surface of the interface, so a refactor can be proved neutral.

The checks in `./dev.sh check` ask whether the app *works*. They cannot see a
page that looks wrong, which is how a deleted stylesheet reached a phone twice.
This is the missing half: capture all nine surfaces, change something, capture
again, and compare.

    python tools/shots.py before
    ...make the change, then: python preview/build_preview.py
    python tools/shots.py after
    python tools/shots.py --compare before after

Two surfaces move on their own - Explore rotates its reel and the player's
scrubber advances - so a difference there is not automatically a regression.
Confirm by capturing the SAME build twice: a real regression shows a difference
that a same-build pair does not.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "preview" / "fam-artifact.html"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

#: name -> the script that puts the interface on that surface.
SURFACES = [
    ("myfam", "setTab('myfam')"),
    ("home", "setTab('home')"),
    ("dailyfam", "openPlayFAM()"),
    ("explore", "openExplore()"),
    ("profile", "openProfile()"),
    ("messages", "openMessages()"),
    ("mixdetail", "setTab('playfam'); document.querySelectorAll('.mix-card')[0].click()"),
    ("mixpicker", "openMixPicker && openMixPicker()"),
    ("player", "setTab('myfam'); document.querySelectorAll('.gd-card')[0].click()"),
]


def capture(out_dir: pathlib.Path) -> int:
    import asyncio

    from playwright.async_api import async_playwright

    out_dir.mkdir(parents=True, exist_ok=True)

    async def run() -> int:
        async with async_playwright() as p:
            kwargs = {"executable_path": CHROME} if pathlib.Path(CHROME).exists() else {}
            browser = await p.chromium.launch(**kwargs)
            page = await browser.new_page(
                viewport={"width": 390, "height": 844}, device_scale_factor=2
            )
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto(PAGE.as_uri())
            await page.wait_for_timeout(1500)
            for name, script in SURFACES:
                try:
                    await page.evaluate(script)
                except Exception as exc:  # a surface that will not open is itself news
                    print(f"  {name:11} could not open: {exc}")
                    errors.append(f"{name}: {exc}")
                    continue
                await page.wait_for_timeout(900)
                target = await page.query_selector(".phone") or await page.query_selector("body")
                (out_dir / f"{name}.png").write_bytes(await target.screenshot())
                print(f"  {name:11} captured")
            await browser.close()
            if errors:
                print("errors:", errors)
            return 1 if errors else 0

    return asyncio.run(run())


def compare(a: pathlib.Path, b: pathlib.Path) -> int:
    from PIL import Image, ImageChops

    differing = 0
    for left in sorted(a.glob("*.png")):
        right = b / left.name
        if not right.exists():
            print(f"  {left.stem:11} missing from {b}")
            differing += 1
            continue
        one, two = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
        if one.size != two.size:
            print(f"  {left.stem:11} SIZE CHANGED {one.size} -> {two.size}")
            differing += 1
            continue
        box = ImageChops.difference(one, two).getbbox()
        if box is None:
            print(f"  {left.stem:11} identical")
        else:
            print(f"  {left.stem:11} DIFFERS in {box}")
            differing += 1
    print(f"\n{differing} surface(s) differ")
    return 1 if differing else 0


def main(argv: list[str]) -> int:
    if not PAGE.exists():
        print("build the preview first: python preview/build_preview.py")
        return 1
    if argv[:1] == ["--compare"]:
        if len(argv) != 3:
            print("usage: python tools/shots.py --compare <before> <after>")
            return 1
        return compare(pathlib.Path(argv[1]), pathlib.Path(argv[2]))
    out = pathlib.Path(argv[0]) if argv else ROOT / "shots"
    return capture(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
