"""A standalone page for looking at the loading screen.

The screen only exists while an episode is being written, which on a cache
hit is 450ms and against fixtures is less. That is correct behaviour and
useless for judging how it looks, so this lifts it out and holds it still.

Everything here is *extracted* from static/index.html rather than copied:
the markup, the CSS, the palette and the fonts all come from the shipped
file at build time. A second hand-written copy would drift, and then the
thing being judged would not be the thing that ships.

    python tools/build_loading_demo.py     ->  preview/loading-screen.html
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "static" / "index.html"
OUT = ROOT / "preview" / "loading-screen.html"


def slice_between(text: str, start: str, end: str, what: str) -> str:
    a = text.find(start)
    if a == -1:
        sys.exit(f"could not find the start of {what} in static/index.html")
    b = text.find(end, a + len(start))
    if b == -1:
        sys.exit(f"could not find the end of {what} in static/index.html")
    return text[a:b]


def build() -> str:
    src = SOURCE.read_text()

    fonts = "\n".join(
        line for line in src.splitlines()
        if "fonts.googleapis.com" in line or "fonts.gstatic.com" in line
    )
    palette = slice_between(src, "  :root{", "  }", "the palette") + "  }"
    css = slice_between(
        src, "  /* -------- the loading screen -------- */", "  .demo-badge{",
        "the loading screen's CSS")
    gen_text = slice_between(src, "  .gen-text{", "  }", ".gen-text") + "  }"
    markup = slice_between(
        src, '      <div class="fam-loading" id="famLoading"', "\n      <!--",
        "the loading screen's markup")

    if "famRise" not in css or "GEN_MIN" in css:
        sys.exit("the extracted CSS does not look like the loading screen")

    # The four states the status line actually takes, in the words the app
    # uses. Kept in step with startHonestWait() and the two notices.
    states = [
        ("Instant answer", "Writing your episode…"),
        ("Needs today's facts", "Answering now — checking sources underneath… 6s"),
        ("No API key", "Playing the built-in sample script…"),
        ("Key rejected", "The server's API key was rejected — no episode can be written."),
    ]
    buttons = "\n".join(
        f'      <button onclick="setStatus({i})">{label}</button>'
        for i, (label, _) in enumerate(states)
    )
    status_js = ",\n".join(f'    {text!r}' for _, text in states)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAM — the loading screen</title>
{fonts}
<style>
{palette}
  *{{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  body{{
    margin:0; min-height:100vh; background:var(--page-bg); color:var(--paper);
    font-family:'Space Grotesk', sans-serif;
    padding:34px 18px 60px; display:flex; flex-direction:column; align-items:center; gap:22px;
  }}
  .lead{{ max-width:430px; text-align:center; }}
  .lead h1{{ font-family:'Fraunces', serif; font-weight:500; font-size:25px;
    color:var(--copper); margin:0 0 8px; }}
  .lead p{{ font-size:12.5px; line-height:1.6; color:var(--text-muted); margin:0; }}
  /* The same 300x620 frame the preview uses, so proportions are honest. */
  .phone{{ width:300px; height:620px; position:relative; }}
  .frame{{
    position:absolute; inset:0; border-radius:34px; overflow:hidden;
    background:var(--bg); border:1px solid var(--border-strong);
    box-shadow:0 30px 70px -24px rgba(0,0,0,0.7);
  }}
  .controls{{ display:flex; flex-wrap:wrap; gap:7px; justify-content:center; max-width:430px; }}
  .controls button{{
    background:var(--surface); color:var(--text); border:1px solid var(--border-strong);
    border-radius:20px; padding:8px 13px; font-family:'JetBrains Mono', monospace;
    font-size:10.5px; cursor:pointer;
  }}
  .controls button:active{{ transform:scale(0.96); }}
  .controls button.on{{ background:var(--copper); color:var(--ink); border-color:var(--copper); }}
  .note{{ max-width:430px; text-align:center; font-family:'JetBrains Mono', monospace;
    font-size:10px; line-height:1.7; color:var(--muted); }}
  .note b{{ color:var(--copper); font-weight:500; }}

{css}
{gen_text}
</style>
</head>
<body>
  <div class="lead">
    <h1>The loading screen</h1>
    <p>Held still so it can be looked at. In the app it is shown the moment a
    search, a tile or a mix is tapped, and cleared the instant the first audio
    arrives — so it lasts as long as the wait does, and no longer.</p>
  </div>

  <div class="phone"><div class="frame">
{markup}  </div></div>

  <div class="controls" id="controls">
{buttons}
      <button onclick="flash()">Show a cache hit (450ms)</button>
  </div>

  <div class="note">
    The line underneath is the honest wait from <b>PROBLEMS.md §55</b>: it says
    what is being waited for, never how good it will be. On a cache hit the
    whole thing lasts 450ms — the floor added because 3ms read as a glitch
    rather than as speed.
  </div>

<script>
  var STATUSES = [
{status_js}
  ];
  var el = document.getElementById("famLoading");
  var line = document.getElementById("famLoadingStatus");
  var buttons = document.getElementById("controls").querySelectorAll("button");

  function setStatus(i){{
    line.textContent = STATUSES[i];
    el.classList.add("active");
    buttons.forEach(function(b, n){{ b.classList.toggle("on", n === i); }});
  }}

  // What a cache hit actually looks like: shown, then held to the floor and
  // cleared. Anything shorter is the flash the floor exists to prevent.
  function flash(){{
    el.classList.remove("active");
    setTimeout(function(){{ el.classList.add("active"); }}, 260);
    setTimeout(function(){{ el.classList.remove("active"); }}, 260 + 450);
  }}

  setStatus(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"{OUT}  ({OUT.stat().st_size // 1024} KB)")
