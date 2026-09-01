"""Build a single self-contained HTML file of the FAM interface.

Why this exists: reviewing the app on a phone used to mean running the server,
being on the same wifi, and finding the laptop's LAN address - or looking at
screenshots, which cannot be tapped. This produces one file with no backend,
no build step and no install, which can be published somewhere and opened on a
phone in one tap.

What it is honest about: it is the *interface*, not the product. There is no
model behind it, so scripts are canned and audio is generated silence at the
right length. Transport controls, progress, swipes, mixes, the topic picker
and the feeds are all real code doing real work against fake data - which is
exactly the layer worth checking on a phone. Anything about writing quality or
time-to-first-audio has to be checked against the real server.

    python preview/build_preview.py            # -> preview/fam-preview.html
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
OUT = ROOT / "preview" / "fam-preview.html"
#: Same page, minus the outer document tags, for publishing as an Artifact.
#: The Artifact host supplies <!doctype>, <head> and <body> itself, so a full
#: document here would nest one inside another and render as text.
OUT_ARTIFACT = ROOT / "preview" / "fam-artifact.html"


def load_fixtures() -> dict:
    """Canned API responses, built from the real bank so the preview shows the
    same topics the app would. Imported rather than duplicated - a fixture that
    drifts from the code is worse than no fixture."""
    sys.path.insert(0, str(ROOT))
    import mixes as mixes_mod
    import topics as topics_mod

    bank = [t.as_dict() for t in topics_mod.TOPIC_BANK]
    by_id = {t["id"]: t for t in bank}

    def section(key, title, ids):
        return {"key": key, "title": title,
                "topics": [by_id[i] for i in ids if i in by_id], "empty_reason": ""}

    myfam = {
        "personalised": True,
        "sections": [
            section(key, title, ids)
            for (key, title), ids in zip(topics_mod.SECTIONS, [
                ["chip-supply", "energy-grid", "founder-motivation", "hormuz"],
                ["stadium-money", "sleep-science", "space-race", "longevity-claims"],
                ["ai-agents", "fed-next-move", "housing-market", "operator-ceos",
                 "hollywood-comebacks", "habits-research"],
            ])
        ],
    }

    def mix(mix_id, name, ids, typed=(), public=False):
        items = [dict(by_id[i], query=by_id[i]["query"], custom=False) for i in ids]
        items += [{"id": "q:" + t.lower().replace(" ", "")[:10], "title": t, "query": t,
                   "custom": True, "subtitle": "Added by you", "icon": "leaf"} for t in typed]
        return {"id": mix_id, "name": name, "items": items,
                "topics": [i for i in items if not i["custom"]],
                "topic_ids": [i["id"] for i in items if not i["custom"]],
                "custom_count": len(typed), "public": public,
                "created_at": 0, "updated_at": 0}

    mixes = {
        "mixes": [
            mix("m1", "Morning", ["fed-next-move", "ai-agents", "morning-mindset"],
                public=True),
            mix("m2", "At the gym", ["training-load", "the-trade", "habits-research"]),
            mix("m3", "Wind down", ["sleep-science", "anxiety-loop"],
                typed=["what my council is doing about the high street"]),
        ],
        "starters": [{"name": n, "topic_ids": list(i)} for n, i in mixes_mod.STARTER_MIXES],
    }

    explore = {"episodes": [
        {"query": q, "title": q[:1].upper() + q[1:], "minutes": m,
         "plays": p, "thread": th, "age_seconds": age,
         "echoed_by": "Rachel Solomon" if m == 5 else ""}
        for q, m, p, th, age in [
            ("why the strait of hormuz moves the oil price", 3, 4,
             "why the shipping lanes run through Omani water", 140),
            ("what habit research actually shows about lasting change", 2, 2,
             "why streaks break in the third week", 900),
            ("how reusable rockets changed the economics of spaceflight", 5, 7,
             "what happens when launch gets cheaper than shipping", 4000),
            ("a good mindset for when I wake up in the morning", 1, 3,
             "why a flat morning cortisol curve might mean burnout", 9000),
            ("what the Federal Reserve is likely to do about interest rates", 3, 9,
             "who actually loses when rates stay high", 30000),
            ("how a city's food supply chain actually works", 4, 1,
             "the three days of stock nobody plans for", 100000),
        ]
    ]}

    import time as _time
    return {
        "/api/myfam": myfam,
        "/api/mixes": mixes,
        "/api/topics": {"topics": bank},
        # The preview never reads a real file: it stands in for the extraction
        # so the flow and the chips can be exercised on a phone.
        "/api/attach": {"id": "preview-attachment", "kind": "document",
                        "name": "Attached file", "chars": 4200, "url": "",
                        "preview": "A stand-in for extracted text."},
        "/api/explore": explore,
        "/api/next": {"thread": "why the shipping lanes run through Omani water"},
        # Two open threads; the shim seeds two part-heard episodes alongside
        # them, so Go Deeper opens as a full grid rather than one lonely card.
        "/api/godeeper": {"threads": [
            {"thread": "why the shipping lanes run through Omani water",
             "title": "The Two-Mile Lane That Moves the Oil",
             "from_title": "Why the Strait of Hormuz Moves the Oil Price", "at": 0},
            {"thread": "how NIL money changed college football recruiting",
             "title": "The New College Football Arms Race",
             "from_title": "Who Really Pays for a Stadium", "at": 0},
        ]},
        "/api/voices": {"voices": [
            {"id": "preview:narrator", "label": "Narrator (preview)", "engine": "preview"}
        ], "default": "preview:narrator"},
        "/api/profile": {
            "listener": "preview-listener", "played": 34, "finished": 21,
            "searched": 12, "open_threads": 2,
            "subjects": ["tech", "money", "science", "health"],
            "since": _time.time() - 63 * 86400,
            "name": "Ian Solomon", "handle": "iansolomon",
            "joined": _time.time() - 63 * 86400,
            "echo_count": 7,
            "mixes": [
                {"id": "m1", "name": "Morning Run", "public": True,
                 "items": [{"id": "x"}] * 14, "topics": [], "topic_ids": [],
                 "custom_count": 0, "created_at": 0, "updated_at": 0},
                {"id": "m2", "name": "Market Watch", "public": True,
                 "items": [{"id": "x"}] * 9, "topics": [], "topic_ids": [],
                 "custom_count": 0, "created_at": 0, "updated_at": 0},
                {"id": "m3", "name": "Kids' Questions", "public": True,
                 "items": [{"id": "x"}] * 21, "topics": [], "topic_ids": [],
                 "custom_count": 0, "created_at": 0, "updated_at": 0},
            ],
            "echoes": [
                {"title": "The Two-Mile Lane That Moves the Oil", "minutes": 4,
                 "query": "why the strait of hormuz moves the oil price", "asked": True},
                {"title": "How a City's Food Supply Chain Works", "minutes": 4,
                 "query": "how a city's food supply chain actually works", "asked": False},
                {"title": "Who Actually Makes the World's Chips", "minutes": 6,
                 "query": "why semiconductor manufacturing is concentrated", "asked": True},
                {"title": "What We Actually Know About Sleep", "minutes": 5,
                 "query": "what sleep research actually establishes", "asked": False},
            ],
        },
        "/api/health": {"ok": True, "demo": True, "engine": "preview"},
        "/api/event": {"ok": True},
    }


SHIM = """
<script>
/* ---- Preview shim -------------------------------------------------------
   Stands in for the Python server so this file works on its own. Every fetch
   the app makes is answered from fixtures baked in at build time; /api/audio
   returns silence of the right length so the transport, the progress bar and
   the reel timing all behave exactly as they do against the real server.
   Nothing here touches the network. -------------------------------------- */
(function () {
  var FIXTURES = __FIXTURES__;
  var SAMPLE_RATE = 22050;
  var realFetch = window.fetch.bind(window);
  var mixes = JSON.parse(JSON.stringify(FIXTURES["/api/mixes"]));
  var nextMixId = 100;

  function json(body, status) {
    return Promise.resolve(new Response(JSON.stringify(body), {
      status: status || 200, headers: { "Content-Type": "application/json" }
    }));
  }

  // Silence, streamed in chunks, so the player's buffering logic runs for real.
  function silence(seconds) {
    var total = Math.round(seconds * SAMPLE_RATE);
    var sent = 0;
    var stream = new ReadableStream({
      pull: function (c) {
        if (sent >= total) { c.close(); return; }
        var n = Math.min(SAMPLE_RATE, total - sent);
        sent += n;
        c.enqueue(new Uint8Array(n * 2));
        return new Promise(function (r) { setTimeout(r, 60); });
      }
    });
    return Promise.resolve(new Response(stream, {
      status: 200,
      headers: { "Content-Type": "audio/L16", "X-Sample-Rate": String(SAMPLE_RATE) }
    }));
  }

  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.indexOf("/api/") === -1) return realFetch(input, init);
    var path = url.split("?")[0];
    var method = ((init && init.method) || "GET").toUpperCase();
    var qs = new URLSearchParams((url.split("?")[1] || ""));

    if (path === "/api/audio") {
      return silence(Math.max(1, Number(qs.get("minutes") || 1)) * 60);
    }
    if (path === "/api/attach") {
      if (method === "DELETE") return json({ ok: true });
      // Echo the name back so the chip reads like the real thing. No file is
      // ever parsed here - the preview has no server.
      var sent = JSON.parse((init && init.body) || "{}");
      var stub = FIXTURES["/api/attach"];
      return json({
        id: "preview-" + Math.random().toString(36).slice(2, 8),
        kind: sent.kind || "document",
        name: sent.name || sent.url || stub.name,
        chars: sent.kind === "image" ? 0 : stub.chars,
        url: sent.url || "", preview: stub.preview
      });
    }
    if (path === "/api/mixes" && method === "GET") return json(mixes);
    if (path === "/api/mixes" && method === "POST") {
      var body = JSON.parse((init && init.body) || "{}");
      var made = buildMix("m" + (nextMixId++), body.name || "New mix",
                          body.topic_ids || [], !!body.public);
      mixes.mixes.push(made);
      return json(made);
    }
    if (path.indexOf("/api/mixes/") === 0) {
      var id = path.split("/").pop();
      var at = mixes.mixes.findIndex(function (m) { return m.id === id; });
      if (at === -1) return json({ error: "That mix no longer exists." }, 404);
      if (method === "DELETE") { mixes.mixes.splice(at, 1); return json({ ok: true }); }
      var patch = JSON.parse((init && init.body) || "{}");
      var current = mixes.mixes[at];
      mixes.mixes[at] = buildMix(id, patch.name || current.name,
        patch.topic_ids !== undefined ? patch.topic_ids : current.items,
        patch.public !== undefined ? patch.public : current.public);
      return json(mixes.mixes[at]);
    }
    if (FIXTURES[path]) return json(FIXTURES[path]);
    return json({ error: "Not available in the preview build." }, 404);
  };

  function buildMix(id, name, entries, isPublic) {
    var bank = {};
    FIXTURES["/api/topics"].topics.forEach(function (t) { bank[t.id] = t; });
    var items = entries.map(function (e) {
      if (typeof e === "string") {
        var t = bank[e];
        return t ? { id: t.id, title: t.title, query: t.query, custom: false,
                     subtitle: t.subtitle, icon: t.icon } : null;
      }
      if (e && e.custom !== undefined) return e;           // already an item
      if (e && e.query) {
        return { id: "q:" + e.query.toLowerCase().slice(0, 24), title: e.title || e.query,
                 query: e.query, custom: true, subtitle: "Added by you", icon: "leaf" };
      }
      return null;
    }).filter(Boolean);
    return { id: id, name: name, items: items,
             topics: items.filter(function (i) { return !i.custom; }),
             topic_ids: items.filter(function (i) { return !i.custom; })
                             .map(function (i) { return i.id; }),
             custom_count: items.filter(function (i) { return i.custom; }).length,
             public: !!isPublic, created_at: 0, updated_at: 0 };
  }

  // Two part-heard episodes, so the preview shows Go Deeper as it looks once
  // someone has been using the app. Seeded once and then left alone, so
  // anything you do to it in the preview sticks.
  try {
    if (!localStorage.getItem("fam_resume")) {
      localStorage.setItem("fam_resume", JSON.stringify({
        "why everyone is talking about AI agents":
          { minutes: 7, at: 259, saved: Date.now() },
        "who actually makes the world's chips":
          { minutes: 6, at: 50, saved: Date.now() - 9000 }
      }));
    }
  } catch (e) { /* private browsing: the section is simply emptier */ }

  // Say what this is, once, without covering anything up.
  window.addEventListener("load", function () {
    var bar = document.createElement("div");
    bar.textContent = "Preview build \\u00b7 interface only \\u00b7 no scripts, no audio";
    // Top, not bottom: at the bottom it sat over the tab bar, which is the
    // one piece of chrome a preview must never obscure.
    bar.style.cssText = "position:fixed;left:0;right:0;top:0;z-index:9999;" +
      "padding:4px 8px;text-align:center;font:600 9px/1.3 monospace;" +
      "letter-spacing:.06em;color:#0d0b12;background:#d4a853;";
    document.body.appendChild(bar);
    setTimeout(function () { bar.style.transition = "opacity .6s"; bar.style.opacity = "0"; }, 6000);
    setTimeout(function () { bar.remove(); }, 7000);
  });
})();
</script>
"""


def build() -> pathlib.Path:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    audio_js = (STATIC / "fam-audio.js").read_text(encoding="utf-8")

    # Inline the player: one file, so it can be opened from anywhere.
    html, n = re.subn(r'<script src="[^"]*fam-audio\.js"[^>]*></script>',
                      "<script>\n" + audio_js + "\n</script>", html)
    if n != 1:
        raise SystemExit("could not inline fam-audio.js - has the script tag changed?")

    shim = SHIM.replace("__FIXTURES__", json.dumps(load_fixtures()))
    # The shim has to be installed before the first line of app code runs, so
    # it goes immediately before the first inline <script> in the document.
    at = html.index("<script>")
    html = html[:at] + shim + html[at:]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


def build_artifact(html: str) -> pathlib.Path:
    """Strip the outer document tags so the Artifact wrapper can supply them.

    Everything else is kept verbatim, including the Google Fonts link - the
    one external host the Artifact CSP admits, and the one this app uses.
    """
    head = re.search(r"<head[^>]*>(.*?)</head>", html, re.S)
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.S)
    if not head or not body:
        raise SystemExit("could not split the document - has the shell changed?")
    head_inner = head.group(1)
    # The host already sets charset and viewport; ours would be duplicates.
    head_inner = re.sub(r"<meta[^>]*charset[^>]*>", "", head_inner)
    head_inner = re.sub(r"<meta[^>]*viewport[^>]*>", "", head_inner)
    OUT_ARTIFACT.write_text(head_inner.strip() + "\n" + body.group(1).strip(),
                            encoding="utf-8")
    return OUT_ARTIFACT


if __name__ == "__main__":
    path = build()
    art = build_artifact(path.read_text(encoding="utf-8"))
    for f in (path, art):
        print(f"{f}  ({f.stat().st_size / 1024:.0f} KB)")
