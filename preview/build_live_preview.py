"""The real interface, on a real database.

`build_preview.py` ships `static/index.html` with every `fetch` answered from
fixtures baked in at build time. That proves layout, flow and interaction on a
phone, and it proves nothing about storage: tap the same tile twice and the
second tap is the same canned JSON as the first.

This builder takes the *same* interface and swaps only that one layer. The
fixture shim becomes a small API implemented against the Artifact `db`
capability, so the app runs against documents that persist across reloads and
are shared by everyone who opens the page. Nothing in `static/index.html` is
modified — the app cannot tell the difference, which is the point: what you are
clicking is the shipped frontend, not a mock of it.

What stays synthetic, because a browser cannot do it:

* **/api/audio** returns silence of the right length, exactly as the fixture
  preview does. There is no Piper and no Claude in a published page, so no
  script is written and no audio is spoken.
* **/api/topics**, **/api/voices** and **/api/health** are reference data, not
  state, and come from the same fixtures as before.

Everything else — the event log, impressions with their section and algo, the
listener table, sessions, accounts, mixes and echoes — is real.

The one honest divergence from the server: the session token lives in
`localStorage` rather than an HttpOnly cookie, because a published page has no
cookie of its own. The shim says so in the badge it renders.

    python preview/build_live_preview.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_preview as bp  # noqa: E402

STATIC = bp.STATIC
OUT = HERE / "fam-live.html"
OUT_ARTIFACT = HERE / "fam-live-artifact.html"

#: Reference data only. Everything stateful is served from the database.
STATIC_PATHS = ("/api/topics", "/api/voices", "/api/health", "/api/attach")


LIVE_SHIM = r"""
<script>
/* ---- Live shim ----------------------------------------------------------
   Stands in for the Python server, the way build_preview's shim does, except
   that the stateful half is a real database rather than fixtures. Every route
   below mirrors the response shape app.py returns, so static/index.html runs
   unmodified and cannot tell which one it is talking to.

   Identity note: the real server keeps the session token in an HttpOnly
   cookie the page cannot read. A published page has no cookie of its own, so
   this keeps it in localStorage instead. That is the only place the demo
   diverges, and the badge says so. ------------------------------------- */
(function () {
  var FIXTURES = __FIXTURES__;
  var SAMPLE_RATE = 22050;
  var ALGO = __ALGO__;
  var realFetch = window.fetch.bind(window);

  // ------------------------------------------------------------ db plumbing
  var db = null, MEM = {}, COLS =
    ["sessions", "accounts", "people", "events", "scripts", "mixes", "echoes"];
  var cache = {}; COLS.forEach(function (c) { cache[c] = []; });
  var UID = "", EMAIL = "", TOKEN = "";
  var badge = null;

  function now() { return Date.now() / 1000; }
  function rid() { return Math.random().toString(36).slice(2, 11); }
  function rows(c) {
    if (db) return cache[c] || [];
    return Object.keys(MEM[c] || {}).map(function (id) {
      var o = { id: id }, d = MEM[c][id];
      for (var k in d) o[k] = d[k];
      return o;
    });
  }
  function put(c, id, data) {
    if (db) return db.doc(c + "/" + id).set(data);
    MEM[c] = MEM[c] || {}; MEM[c][id] = data; paint();
    return Promise.resolve();
  }
  function addDoc(c, data) { return put(c, rid(), data); }
  function getOne(c, id) {
    if (db) return db.doc(c + "/" + id).get().then(function (s) { return s.exists ? s.data() : null; });
    return Promise.resolve((MEM[c] || {})[id] || null);
  }
  function del(c, id) {
    if (db) return db.doc(c + "/" + id).delete();
    if (MEM[c]) delete MEM[c][id];
    paint(); return Promise.resolve();
  }

  // ---------------------------------------------------------- the topic bank
  // The app decides what to say about audio from /api/health. The fixture body
  // has no `mode`, so it fell through to "Audio server unreachable - start it
  // with ./run.sh" - true of a server, and useless advice inside a published
  // page. `demo` is the honest state here: there is no key, so no script is
  // written and nothing is spoken.
  FIXTURES["/api/health"] = {
    ok: true, mode: "demo", demo: true, engine: "prototype",
    credentials: { state: "absent" }, research_words: [],
    sample_rate: SAMPLE_RATE, min_minutes: 1, max_minutes: 10
  };

  var BANK = FIXTURES["/api/topics"].topics;
  var BY_ID = {}; BANK.forEach(function (t) { BY_ID[t.id] = t; });

  // cache.normalize_query, in JS: lowercase, strip punctuation, drop filler,
  // sort what is left — so two phrasings are one row and one payment.
  var FILLER = ("a an the of for on in to and is are was were please give me tell about " +
    "what whats who how why can you i want would like do does explain describe " +
    "summarize summarise recap briefing podcast episode some any there this that with").split(" ");
  function normalize(q) {
    var seen = {}, out = [];
    String(q).toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).forEach(function (t) {
      if (t && FILLER.indexOf(t) === -1 && !seen[t]) { seen[t] = 1; out.push(t); }
    });
    return out.sort().join(" ");
  }
  function keyFor(q, m) { return normalize(q).replace(/\s+/g, "-").slice(0, 60) + "--" + m + "m"; }

  // ------------------------------------------------------------- the reads
  // topics.EventStore.for_user: behavioural rows only. Impressions are
  // excluded here and nowhere else, which is what stops the feed training on
  // what it already showed.
  function behavioural(uid) {
    return rows("events").filter(function (e) {
      return e.user_id === uid && e.kind !== "impression";
    });
  }
  var WEIGHT = { search: 1.0, play: 1.0, complete: 2.5, skip: -1.5 };
  function taste(uid) {
    var s = {}, t = now(), HALF = 14 * 86400;
    behavioural(uid).forEach(function (e) {
      var w = (WEIGHT[e.kind] || 0) * Math.pow(0.5, Math.max(0, t - e.at) / HALF);
      (e.tags ? String(e.tags).split(",") : []).forEach(function (g) {
        if (g) s[g] = (s[g] || 0) + w;
      });
    });
    return s;
  }
  function playedIds(uid) {
    var s = {};
    behavioural(uid).forEach(function (e) {
      if (e.topic_id && (e.kind === "play" || e.kind === "complete")) s[e.topic_id] = 1;
    });
    return s;
  }

  function feed() {
    var profile = taste(UID), mine = playedIds(UID), used = {}, out = {};
    function take(list, n) {
      var got = [];
      for (var i = 0; i < list.length && got.length < n; i++) {
        var t = list[i];
        if (t && !used[t.id] && !mine[t.id]) { used[t.id] = 1; got.push(t); }
      }
      return got;
    }
    var scored = BANK.map(function (t) {
      var s = 0; (t.tags || []).forEach(function (g) { s += (profile[g] || 0); });
      return { t: t, s: s / Math.sqrt((t.tags || []).length || 1) };
    }).filter(function (x) { return x.s > 0; }).sort(function (a, b) { return b.s - a.s; });

    var counts = {};
    rows("events").forEach(function (e) {
      if (e.user_id !== UID && e.topic_id && (e.kind === "play" || e.kind === "complete")) {
        counts[e.topic_id] = (counts[e.topic_id] || 0) + 1;
      }
    });
    var byCount = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; })
      .map(function (id) { return BY_ID[id]; }).filter(Boolean);

    // Filled most-constrained first, exactly as build_feed does: the personal
    // sections choose before the generic ones can claim the bank.
    out.from_history = take(scored.map(function (x) { return x.t; }), 6);
    out.followers = take(byCount, 6);
    out.trending = take(byCount.concat(BANK), 6);
    return { picked: out, personalised: Object.keys(profile).length > 0 };
  }

  var SECTIONS = [
    ["from_history", "Made for you", "Your first episode starts this one off."],
    ["followers", "Your circle is on this", "Nobody you overlap with has listened yet."],
    ["trending", "What FAM can't stop playing", "Nothing has been played yet."]
  ];

  function myfamBody() {
    var f = feed(), shown = [];
    var sections = SECTIONS.map(function (s) {
      var list = f.picked[s[0]] || [];
      list.forEach(function (t) { shown.push({ id: t.id, section: s[0] }); });
      return { key: s[0], title: s[1], topics: list, empty_reason: list.length ? "" : s[2] };
    });
    // One impression row per tile, carrying its shelf and the ranking version.
    if (UID && shown.length) {
      shown.forEach(function (x) {
        addDoc("events", {
          user_id: UID, kind: "impression", topic_id: x.id, text: "",
          tags: ((BY_ID[x.id] || {}).tags || []).join(","), at: now(),
          thread: "", section: x.section, algo: ALGO
        });
      });
    }
    return { sections: sections, personalised: f.personalised, algo: ALGO };
  }

  function profileBody() {
    var beh = behavioural(UID), t = taste(UID);
    var person = rows("people").filter(function (p) { return p.id === UID; })[0] || {};
    var c = { play: 0, complete: 0, search: 0 };
    beh.forEach(function (e) { if (c[e.kind] !== undefined) c[e.kind]++; });
    var myEchoes = rows("echoes").filter(function (e) { return e.user_id === UID; });
    return {
      listener: UID,
      played: c.play, finished: c.complete, searched: c.search,
      open_threads: threads().length,
      subjects: Object.keys(t).filter(function (k) { return t[k] > 0; })
        .sort(function (a, b) { return t[b] - t[a]; }).slice(0, 5),
      since: beh.length ? Math.min.apply(null, beh.map(function (e) { return e.at; })) : 0,
      name: person.name || "", handle: person.handle || "",
      joined: person.joined || 0, last_seen: person.last_seen || 0,
      known: !!person.id,
      echo_count: myEchoes.length,
      mixes: rows("mixes").filter(function (m) { return m.user_id === UID && m.public; })
        .map(shapeMix),
      echoes: myEchoes.slice(0, 12).map(function (e) {
        return { id: e.id, query: e.query, title: e.title, minutes: e.minutes,
                 thread: e.thread || "", at: e.at,
                 by: person.name || "", handle: person.handle || "" };
      })
    };
  }

  // Go Deeper: the follow-up each finished episode left behind.
  function threads() {
    var seen = {}, out = [];
    behavioural(UID).filter(function (e) { return e.kind === "complete" && e.thread; })
      .sort(function (a, b) { return b.at - a.at; })
      .forEach(function (e) {
        if (seen[e.thread]) return; seen[e.thread] = 1;
        var t = BY_ID[e.topic_id];
        out.push({ thread: e.thread, title: e.thread,
                   from_title: t ? t.title : (e.text || "an episode"), at: e.at });
      });
    return out.slice(0, 8);
  }

  function exploreBody(limit) {
    var labels = {};
    rows("echoes").forEach(function (e) {
      if (e.user_id === UID) return;
      var p = rows("people").filter(function (x) { return x.id === e.user_id; })[0];
      labels[e.query] = (p && p.name) || "Someone";
    });
    var eps = rows("scripts").filter(function (s) { return s.expires > now(); })
      .sort(function (a, b) { return b.created - a.created; })
      .slice(0, limit || 30)
      .map(function (s) {
        return {
          query: s.query,
          title: String(s.query).charAt(0).toUpperCase() + String(s.query).slice(1),
          minutes: s.minutes, plays: s.hits || 0, thread: s.thread || "",
          age_seconds: Math.max(0, now() - s.created),
          echoed_by: labels[s.query] || ""
        };
      });
    return { episodes: eps };
  }

  function shapeMix(m) {
    var ids = String(m.items || "").split(",").filter(Boolean);
    var items = ids.map(function (id) {
      var t = BY_ID[id];
      return t ? { id: t.id, title: t.title, query: t.query, custom: false,
                   subtitle: t.subtitle, icon: t.icon }
               : { id: id, title: id.replace(/^q:/, ""), query: id.replace(/^q:/, ""),
                   custom: true, subtitle: "Added by you", icon: "leaf" };
    });
    return { id: m.id, name: m.name, items: items,
             topics: items.filter(function (i) { return !i.custom; }),
             topic_ids: items.filter(function (i) { return !i.custom; }).map(function (i) { return i.id; }),
             custom_count: items.filter(function (i) { return i.custom; }).length,
             public: !!m.public, created_at: m.created_at || 0, updated_at: m.updated_at || 0 };
  }

  // ---------------------------------------------------------------- session
  function mint(forUid) {
    var token = "t_" + rid() + rid();
    var uid = forUid || ("anon_" + rid() + rid());
    return put("sessions", token, {
      user_id: uid, created: now(), expires: now() + 90 * 86400, last_used: now()
    }).then(function () {
      TOKEN = token; UID = uid;
      try { localStorage.setItem("fam_live_session", token); } catch (e) {}
      return getOne("people", uid).then(function (p) {
        return put("people", uid, {
          name: (p && p.name) || "", handle: (p && p.handle) || "",
          joined: (p && p.joined) || now(), last_seen: now()
        });
      });
    });
  }
  function resume() {
    var t = null;
    try { t = localStorage.getItem("fam_live_session"); } catch (e) {}
    if (!t) return mint(null);
    return getOne("sessions", t).then(function (s) {
      if (!s || s.expires < now()) return mint(null);
      TOKEN = t; UID = s.user_id;
      return getOne("accounts", UID).then(function (a) {
        EMAIL = a ? a.email : "";
        return getOne("people", UID).then(function (p) {
          return put("people", UID, {
            name: (p && p.name) || "", handle: (p && p.handle) || "",
            joined: (p && p.joined) || now(), last_seen: now()
          });
        });
      });
    });
  }

  // --------------------------------------------------------------- the wire
  function json(body, status) {
    return new Response(JSON.stringify(body), {
      status: status || 200, headers: { "Content-Type": "application/json" }
    });
  }
  function silence(seconds) {
    var total = Math.round(seconds * SAMPLE_RATE), sent = 0;
    return Promise.resolve(new Response(new ReadableStream({
      pull: function (c) {
        if (sent >= total) { c.close(); return; }
        var n = Math.min(SAMPLE_RATE, total - sent); sent += n;
        c.enqueue(new Uint8Array(n * 2));
        return new Promise(function (r) { setTimeout(r, 60); });
      }
    }), { status: 200, headers: { "Content-Type": "audio/L16", "X-Sample-Rate": String(SAMPLE_RATE) } }));
  }

  var READY = null;   // resolves once storage and the session are settled

  function handle(path, method, qs, body) {
    // ---- identity
    if (path === "/api/auth/me") {
      return json({ user_id: UID, email: EMAIL, authenticated: !!EMAIL });
    }
    if (path === "/api/auth/signup") {
      var em = String(body.email || "").trim().toLowerCase();
      if (!/^[^@\s]+@[^@\s]+\.[^@\s.]{2,}$/.test(em)) return json({ error: "That does not look like an email address." }, 400);
      if (rows("accounts").some(function (a) { return a.email === em; })) return json({ error: "That email is already registered." }, 400);
      if (rows("accounts").some(function (a) { return a.id === UID; })) return json({ error: "This listener already has an account. Log out first." }, 400);
      // Attaches to the id this listener already has: same user_id, so the
      // events, mixes and echoes above are simply theirs now.
      return put("accounts", UID, {
        email: em, password: "(scrypt hash - this prototype stores no credential)",
        created: now(), last_login: now()
      }).then(function () {
        EMAIL = em; paint();
        return json({ user_id: UID, email: em, authenticated: true });
      });
    }
    if (path === "/api/auth/login") {
      var e2 = String(body.email || "").trim().toLowerCase();
      var acct = rows("accounts").filter(function (a) { return a.email === e2; })[0];
      if (!acct) return json({ error: "That email and password do not match." }, 401);
      var old = TOKEN;
      return mint(acct.id).then(function () {
        if (old) del("sessions", old);       // a fresh token, never the old one
        EMAIL = acct.email; paint();
        return json({ user_id: UID, email: EMAIL, authenticated: true });
      });
    }
    if (path === "/api/auth/logout") {
      var gone = TOKEN;
      EMAIL = "";
      try { localStorage.removeItem("fam_live_session"); } catch (e) {}
      return mint(null).then(function () {
        if (gone) del("sessions", gone);
        paint();
        return json({ ok: true });
      });
    }
    if (path === "/api/auth/password") return json({ ok: true });

    // ---- the surfaces
    if (path === "/api/myfam") return json(myfamBody());
    if (path === "/api/profile") return json(profileBody());
    if (path === "/api/godeeper") return json({ threads: threads() });
    if (path === "/api/explore") return json(exploreBody(Number(qs.get("limit") || 30)));
    if (path === "/api/next") {
      var t = threads()[0];
      return json({ thread: t ? t.thread : "" });
    }

    if (path === "/api/event") {
      var tags = body.topic_id && BY_ID[body.topic_id]
        ? (BY_ID[body.topic_id].tags || []).join(",") : "";
      return addDoc("events", {
        user_id: UID, kind: body.kind, topic_id: body.topic_id || "",
        text: body.text || "", tags: tags, at: now(), thread: body.thread || "",
        section: "", algo: ""
      }).then(function () { paint(); return json({ ok: true }); });
    }

    if (path === "/api/me") {
      var nm = String(body.name || "").trim().slice(0, 40);
      var hd = String(body.handle || "").trim().replace(/^@/, "").toLowerCase().slice(0, 24);
      if (!nm) return json({ error: "Give yourself a name." }, 400);
      if (!/^[a-z0-9_.]{2,24}$/.test(hd)) return json({ error: "A handle is 2-24 letters, numbers, dots or underscores." }, 400);
      if (rows("people").some(function (p) { return p.handle === hd && p.id !== UID; })) {
        return json({ error: "@" + hd + " is taken." }, 400);
      }
      var was = rows("people").filter(function (p) { return p.id === UID; })[0] || {};
      return put("people", UID, {
        name: nm, handle: hd, joined: was.joined || now(), last_seen: now()
      }).then(function () { paint(); return json({ user_id: UID, name: nm, handle: hd }); });
    }

    if (path === "/api/echo" && method === "DELETE") {
      var q = qs.get("q") || "", mins = Number(qs.get("minutes") || 3);
      var found = rows("echoes").filter(function (e) {
        return e.user_id === UID && e.query === q && e.minutes === mins;
      })[0];
      if (found) return del("echoes", found.id).then(function () { paint(); return json({ ok: true }); });
      return json({ ok: false });
    }
    if (path === "/api/echo") {
      var already = rows("echoes").filter(function (e) {
        return e.user_id === UID && e.query === body.query && e.minutes === body.minutes;
      })[0];
      var id = already ? already.id : rid();
      return put("echoes", id, {
        user_id: UID, query: body.query, title: body.title || "",
        minutes: body.minutes || 3, thread: body.thread || "", at: now()
      }).then(function () { paint(); return json({ id: id, query: body.query, at: now() }); });
    }

    if (path === "/api/mixes" && method === "GET") {
      return json({
        mixes: rows("mixes").filter(function (m) { return m.user_id === UID; }).map(shapeMix),
        starters: FIXTURES["/api/mixes"].starters
      });
    }
    if (path === "/api/mixes" && method === "POST") {
      var mid = rid();
      return put("mixes", mid, {
        user_id: UID, name: body.name || "New mix",
        items: (body.topic_ids || []).map(function (e) {
          return typeof e === "string" ? e : ("q:" + String(e.query || "").toLowerCase().slice(0, 24));
        }).join(","),
        created_at: now(), updated_at: now(), public: 0
      }).then(function () {
        paint();
        return json(shapeMix(rows("mixes").filter(function (m) { return m.id === mid; })[0]));
      });
    }
    if (path.indexOf("/api/mixes/") === 0) {
      var mixId = path.split("/").pop();
      var cur = rows("mixes").filter(function (m) { return m.id === mixId && m.user_id === UID; })[0];
      if (!cur) return json({ error: "That mix no longer exists." }, 404);
      if (method === "DELETE") return del("mixes", mixId).then(function () { paint(); return json({ ok: true }); });
      return put("mixes", mixId, {
        user_id: UID,
        name: body.name !== undefined ? body.name : cur.name,
        items: body.topic_ids !== undefined
          ? body.topic_ids.map(function (e) {
              return typeof e === "string" ? e : ("q:" + String(e.query || "").toLowerCase().slice(0, 24));
            }).join(",")
          : cur.items,
        created_at: cur.created_at, updated_at: now(),
        public: body.public !== undefined ? (body.public ? 1 : 0) : cur.public
      }).then(function () {
        paint();
        return json(shapeMix(rows("mixes").filter(function (m) { return m.id === mixId; })[0]));
      });
    }

    if (FIXTURES[path]) return json(FIXTURES[path]);
    return json({ error: "Not available in this build." }, 404);
  }

  // The script cache, written on the generation path exactly as the pipeline
  // does: look the key up, count a hit, or write the row on a miss.
  function touchScript(query, minutes) {
    var key = keyFor(query, minutes);
    return getOne("scripts", key).then(function (hit) {
      var live = hit && hit.expires > now();
      return put("scripts", key, live ? {
        query: hit.query, minutes: hit.minutes, sentences: hit.sentences,
        created: hit.created, expires: hit.expires, hits: (hit.hits || 0) + 1,
        thread: hit.thread || ""
      } : {
        query: query, minutes: minutes,
        sentences: "(no model call in this build)",
        created: now(), expires: now() + 86400, hits: 1,
        thread: "what that changes next"
      }).then(function () { paint(); return live; });
    });
  }

  var STATIC_SET = {};
  __STATIC_PATHS__.forEach(function (p) { STATIC_SET[p] = 1; });

  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.indexOf("/api/") === -1) return realFetch(input, init);
    var path = url.split("?")[0];
    var method = ((init && init.method) || "GET").toUpperCase();
    var qs = new URLSearchParams(url.split("?")[1] || "");
    var body = {};
    try { body = JSON.parse((init && init.body) || "{}"); } catch (e) {}

    if (path === "/api/audio") {
      var mins = Math.max(1, Number(qs.get("minutes") || 1));
      var q = qs.get("q") || "";
      // The play is recorded server-side in the real app, from the audio
      // request, because a play that reached the server is a fact.
      READY.then(function () {
        touchScript(q, mins);
        var tid = qs.get("topic_id") || "";
        addDoc("events", {
          user_id: UID, kind: "play", topic_id: tid, text: q,
          tags: tid && BY_ID[tid] ? (BY_ID[tid].tags || []).join(",") : "",
          at: now(), thread: "", section: "", algo: ""
        }).then(paint);
      });
      return silence(mins * 60);
    }
    if (STATIC_SET[path] && path !== "/api/attach") return Promise.resolve(json(FIXTURES[path]));
    if (path === "/api/attach") {
      if (method === "DELETE") return Promise.resolve(json({ ok: true }));
      var stub = FIXTURES["/api/attach"];
      return Promise.resolve(json({
        id: "live-" + rid(), kind: body.kind || "document",
        name: body.name || body.url || stub.name,
        chars: body.kind === "image" ? 0 : stub.chars,
        url: body.url || "", preview: stub.preview
      }));
    }
    return READY.then(function () {
      return Promise.resolve(handle(path, method, qs, body));
    });
  };

  // ----------------------------------------------------------------- badge
  // Small, out of the way, and honest about what this is. It also carries the
  // reset, because a shared database with no way to clear it is a trap.
  function paint() {
    if (!badge) return;
    var ev = rows("events"), imp = ev.filter(function (e) { return e.kind === "impression"; }).length;
    var sc = rows("scripts");
    var plays = sc.reduce(function (a, s) { return a + (s.hits || 0); }, 0);
    var hits = Math.max(0, plays - sc.length);
    badge.querySelector("b").textContent =
      (db ? "live db" : "in memory") + " · " + ev.length + " events (" + imp + " imp) · " +
      sc.length + " scripts · " + (plays ? Math.round(hits / plays * 100) : 0) + "% hit";
  }
  function makeBadge() {
    badge = document.createElement("div");
    badge.style.cssText = "position:fixed;right:8px;top:8px;z-index:99999;display:flex;gap:6px;" +
      "align-items:center;font:500 9.5px/1.4 ui-monospace,monospace;color:#e9bc63;" +
      "background:rgba(12,10,18,.9);border:1px solid #3a3348;border-radius:6px;padding:4px 7px;";
    badge.innerHTML = '<b style="font-weight:500"></b>' +
      '<button style="font:inherit;color:#a79eba;background:#262233;border:1px solid #3a3348;' +
      'border-radius:4px;padding:2px 5px;cursor:pointer">reset</button>';
    badge.querySelector("button").onclick = function () {
      if (!confirm("Delete every document this prototype has written?")) return;
      var jobs = [];
      COLS.forEach(function (c) { rows(c).forEach(function (r) { jobs.push(del(c, r.id)); }); });
      Promise.all(jobs).then(function () {
        try { localStorage.removeItem("fam_live_session"); localStorage.removeItem("fam_resume"); } catch (e) {}
        location.reload();
      });
    };
    document.body.appendChild(badge);
    paint();
    // The app's own DEMO MODE sentence tells you to add a key to .env and
    // restart. Accurate for a server, meaningless in a published page, so the
    // shim states what is actually true of this build instead. Left to the
    // app's own element and styling - only the words change.
    setTimeout(function () {
      var el = document.getElementById("audioMode");
      if (!el) return;
      el.style.color = "var(--copper)";
      el.textContent = "PROTOTYPE \u00b7 the database is real and every tap is stored; " +
        "there is no model and no speech engine here, so playback is silence of the " +
        "right length and no script is written.";
      var canned = document.getElementById("famLoadingStatus");
      if (canned) canned.textContent = "Prototype \u2014 no script is being written\u2026";
    }, 1200);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", makeBadge);
  } else { makeBadge(); }

  // ------------------------------------------------------------------ seed
  // The equivalent of tools/seed_demo.py, and there for the same reason it is:
  // Explore replays other listeners' episodes and by design cannot generate
  // one, so on an empty database it stays blank however much you tap it. Runs
  // once, only when the store is genuinely empty, so a real visit is never
  // overwritten.
  function seedIfEmpty() {
    if (rows("scripts").length || rows("events").length) return Promise.resolve();
    var people = [["anon_seed_rachel", "Rachel", "rachel"],
                  ["anon_seed_sam", "Sam", "sam"],
                  ["anon_seed_priya", "Priya", "priya"]];
    var picks = BANK.slice(0, 8);
    var jobs = [];
    people.forEach(function (p, i) {
      jobs.push(put("people", p[0], {
        name: p[1], handle: p[2], joined: now() - 86400 * 40, last_seen: now() - 3600 * (i + 1)
      }));
      picks.slice(i, i + 4).forEach(function (t, j) {
        jobs.push(addDoc("events", {
          user_id: p[0], kind: "complete", topic_id: t.id, text: t.query,
          tags: (t.tags || []).join(","), at: now() - 3600 * (2 + i * 3 + j),
          thread: "", section: "", algo: ""
        }));
        jobs.push(put("scripts", keyFor(t.query, 3), {
          query: t.query, minutes: 3, sentences: "(seeded — no model call)",
          created: now() - 3600 * (1 + j), expires: now() + 86400,
          hits: 1 + ((i + j) % 4), thread: "what that changes next"
        }));
      });
    });
    // One echo, so an Explore card carries a name rather than "someone asked".
    jobs.push(addDoc("echoes", {
      user_id: "anon_seed_rachel", query: picks[0].query, title: picks[0].title,
      minutes: 3, thread: "", at: now() - 1800
    }));
    return Promise.all(jobs);
  }

  // Two mixes and a finished episode for whoever is looking, so DailyFAM,
  // Go Deeper and the profile are not empty on a first visit either.
  function seedMine() {
    if (rows("mixes").some(function (m) { return m.user_id === UID; })) return Promise.resolve();
    var jobs = [];
    jobs.push(put("mixes", rid(), {
      user_id: UID, name: "Morning Run",
      items: BANK.slice(0, 4).map(function (t) { return t.id; }).join(","),
      created_at: now() - 86400, updated_at: now() - 86400, public: 1
    }));
    jobs.push(put("mixes", rid(), {
      user_id: UID, name: "Market Watch",
      items: BANK.slice(4, 7).map(function (t) { return t.id; }).join(","),
      created_at: now() - 3600, updated_at: now() - 3600, public: 0
    }));
    var t0 = BANK[2];
    jobs.push(addDoc("events", {
      user_id: UID, kind: "complete", topic_id: t0.id, text: t0.query,
      tags: (t0.tags || []).join(","), at: now() - 7200,
      thread: "how those agents are actually evaluated", section: "", algo: ""
    }));
    jobs.push(addDoc("echoes", {
      user_id: UID, query: t0.query, title: t0.title, minutes: 3, thread: "", at: now() - 5400
    }));
    return Promise.all(jobs);
  }

  // ------------------------------------------------------------- bootstrap
  READY = new Promise(function (resolve) {
    var settle = function () {
      seedIfEmpty().then(seedMine).then(function () { paint(); resolve(); })
        .catch(function () { paint(); resolve(); });
    };
    if (!window.claude || !window.claude.use) return resume().then(settle);
    window.claude.use("db").then(function (ns) {
      if (!ns) return resume().then(settle);        // no storage: run in memory
      db = ns;
      var pending = COLS.length;
      COLS.forEach(function (c) {
        db.collection(c).onSnapshot(function (snap) {
          cache[c] = snap.docs.map(function (d) {
            var o = { id: d.id }, data = d.data() || {};
            for (var k in data) o[k] = data[k];
            return o;
          });
          paint();
          if (pending > 0 && --pending === 0) resume().then(settle);
        }, function () { if (pending > 0 && --pending === 0) resume().then(settle); });
      });
      // Never hang the app on storage that does not answer.
      setTimeout(function () { if (pending > 0) { pending = 0; resume().then(settle); } }, 6000);
    }).catch(function () { resume().then(settle); });
  });
})();
</script>
"""


def build() -> pathlib.Path:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    audio_js = (STATIC / "fam-audio.js").read_text(encoding="utf-8")

    html, n = re.subn(r'<script src="[^"]*fam-audio\.js"[^>]*></script>',
                      "<script>\n" + audio_js + "\n</script>", html)
    if n != 1:
        raise SystemExit("could not inline fam-audio.js - has the script tag changed?")

    # topics.ALGO_VERSION, so an impression row carries the same stamp the
    # server would write rather than a number invented here.
    sys.path.insert(0, str(HERE.parent))
    import topics  # noqa: E402

    shim = (LIVE_SHIM
            .replace("__FIXTURES__", json.dumps(bp.load_fixtures()))
            .replace("__ALGO__", json.dumps(topics.ALGO_VERSION))
            .replace("__STATIC_PATHS__", json.dumps(list(STATIC_PATHS))))

    at = html.index("<script>")
    html = html[:at] + shim + html[at:]
    OUT.write_text(html, encoding="utf-8")
    return OUT


def build_artifact(html: str) -> pathlib.Path:
    head = re.search(r"<head[^>]*>(.*?)</head>", html, re.S)
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.S)
    if not head or not body:
        raise SystemExit("could not split the document - has the shell changed?")
    head_inner = re.sub(r"<meta[^>]*charset[^>]*>", "", head.group(1))
    head_inner = re.sub(r"<meta[^>]*viewport[^>]*>", "", head_inner)
    OUT_ARTIFACT.write_text(head_inner.strip() + "\n" + body.group(1).strip(), encoding="utf-8")
    return OUT_ARTIFACT


if __name__ == "__main__":
    path = build()
    art = build_artifact(path.read_text(encoding="utf-8"))
    for f in (path, art):
        print(f"{f}  ({f.stat().st_size / 1024:.0f} KB)")
