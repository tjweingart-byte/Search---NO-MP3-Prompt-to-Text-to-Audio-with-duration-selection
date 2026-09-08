"""Measure whether the browser's main thread stalls while an episode streams.

The report this exists for: 3-5 seconds of audio, then 30-45 seconds of silence
during which nothing on the page can be clicked and Chrome offers "Page
Unresponsive", then playback resumes exactly where it stopped.

"Page Unresponsive" is the tell. Audio starving would be silence with a live
page; a dead page means the main thread is blocked, and audio already scheduled
on the audio thread keeps playing until it runs out - which is the 3-5 seconds.

This needs no API key. The stall is a function of how many bytes arrive and how
fast, not of what the words are, so the built-in sample script reproduces it.

    python tools/stall_probe.py             # 3-minute episode
    python tools/stall_probe.py --minutes 5

It plants a heartbeat on a 50 ms timer and reports the largest gap between
beats. A responsive page keeps that under ~0.2s; anything above a second is the
page being unable to schedule audio or answer a click.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = int(os.environ.get("STALL_PROBE_PORT", "8077"))

HEARTBEAT = """
() => {
  window.__beats = [];
  window.__lastBeat = performance.now();
  window.__maxGap = 0;
  setInterval(() => {
    const now = performance.now();
    const gap = now - window.__lastBeat;
    if (gap > window.__maxGap) window.__maxGap = gap;
    window.__lastBeat = now;
    window.__beats.push(gap);
  }, 50);
}
"""


def _launch(pw):
    """Same fallback as tools/smoke_preview.py: this environment ships a
    Chromium at a different version than the Playwright package expects."""
    import pathlib

    args = ["--autoplay-policy=no-user-gesture-required"]
    try:
        return pw.chromium.launch(args=args)
    except Exception as first:
        for pattern in ("opt/pw-browsers/chromium-*/chrome-linux/chrome",
                        "opt/pw-browsers/chromium/chrome-linux/chrome"):
            for path in sorted(pathlib.Path("/").glob(pattern)):
                try:
                    return pw.chromium.launch(executable_path=str(path), args=args)
                except Exception:
                    continue
        raise first


def run(minutes: int, seconds_to_watch: float) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:  pip install playwright", file=sys.stderr)
        return 2

    env = dict(os.environ, PORT=str(PORT), FAM_IGNORE_DOTENV="1")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(60):
            try:
                import urllib.request

                urllib.request.urlopen(f"{base}/api/health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("server did not start", file=sys.stderr)
            return 2

        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_page()
            page.goto(base, wait_until="load")
            page.evaluate(HEARTBEAT)

            print(f"playing a {minutes}-minute episode and watching the main thread…")
            started = time.time()
            page.evaluate(
                """([q, m]) => window.FamAudio.play(q, m, {}, "", "", {})""",
                ["what habit research actually shows about lasting change", minutes],
            )

            # Poll from the driver, not the page: if the page is blocked, these
            # calls queue up and the gap shows in the heartbeat afterwards.
            first_audio_at = None
            deadline = started + seconds_to_watch
            while time.time() < deadline:
                time.sleep(0.5)
                try:
                    got = page.evaluate("() => window.FamAudio.duration()")
                except Exception:
                    continue
                if first_audio_at is None and got > 0:
                    first_audio_at = time.time() - started

            gaps = page.evaluate("() => window.__beats || []")
            max_gap = page.evaluate("() => window.__maxGap || 0") / 1000.0
            buffered = page.evaluate("() => window.FamAudio.duration()")
            played = page.evaluate("() => window.FamAudio.position()")
            browser.close()

        over_a_second = [g / 1000.0 for g in gaps if g > 1000]
        print()
        print(f"  first audio            {first_audio_at or 0:.2f}s")
        print(f"  audio buffered         {buffered:.1f}s of {minutes * 60}s")
        print(f"  audio actually played  {played:.1f}s")
        print(f"  longest main-thread stall  {max_gap:.2f}s")
        if over_a_second:
            print(f"  stalls over 1s: {len(over_a_second)}  "
                  f"({', '.join(f'{g:.1f}s' for g in over_a_second[:8])})")
        print()
        if max_gap > 1.0:
            print("STALLED. The page could not schedule audio or answer a click for "
                  f"{max_gap:.1f}s.")
            return 1
        print("No stall: the main thread stayed responsive throughout.")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--watch", type=float, default=25.0,
                    help="seconds to observe before reporting")
    args = ap.parse_args()
    return run(args.minutes, args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
