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
  var TAG_WORDS = __TAG_WORDS__;   // topics.TAG_WORDS, verbatim
  var TAG_LABELS = __TAG_LABELS__; // topics.TAG_LABELS, verbatim
  var LANGUAGES = __LANGUAGES__;   // preferences.LANGUAGES, verbatim
  var MAX_INTERESTS = __MAX_INTERESTS__;
  var INTEREST_WEIGHT = __INTEREST_WEIGHT__;
  var VOLATILE = __VOLATILE__;     // cache.research_words(), verbatim
  var NEAR = __NEAR__;             // the shipped CACHE_VECTOR thresholds
  var realFetch = window.fetch.bind(window);

  // ------------------------------------------------------------ db plumbing
  var db = null, MEM = {}, COLS =
    ["sessions", "accounts", "people", "events", "scripts", "mixes", "echoes",
     "prefs"];
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

  // ------------------------------------------------- near matching, in the browser
  // A port of embeddings.py and cache.comparable, so the panel can show the
  // near-match cache deciding rather than describe it. Same features, same
  // constants, same guards, same order.
  //
  // One honest difference: Python hashes features with blake2b and this uses a
  // 32-bit FNV-1a, because a browser has no blake2b and pulling one in for a
  // prototype is not worth it. Both are signed hashing over the same feature
  // set, so which pairs match agrees; the score can differ in the third
  // decimal. Nothing compares a vector across the two - the browser store is
  // its own - so the difference cannot produce a wrong answer, only a
  // slightly different number on screen.
  var DIMS = 256, ORDER = 3, NGRAM_WEIGHT = 0.15;
  var NUMERALS = {
    zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5", six: "6",
    seven: "7", eight: "8", nine: "9", ten: "10", eleven: "11", twelve: "12",
    thirteen: "13", fourteen: "14", fifteen: "15", sixteen: "16",
    seventeen: "17", eighteen: "18", nineteen: "19", twenty: "20",
    first: "1", second: "2", third: "3", fourth: "4", fifth: "5", sixth: "6",
    seventh: "7", eighth: "8", ninth: "9", tenth: "10"
  };
  function tokensOf(text) {
    var out = [];
    String(text).toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).forEach(function (t) {
      if (t) out.push(NUMERALS[t] || t);
    });
    return out;
  }
  function numbersOf(text) {
    var set = {};
    tokensOf(text).forEach(function (t) { if (/^\d+$/.test(t)) set[t] = 1; });
    return Object.keys(set).sort().join(",");
  }
  function slotSign(feature) {
    var h = 2166136261;
    for (var i = 0; i < feature.length; i++) {
      h ^= feature.charCodeAt(i);
      h = (h + (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24)) >>> 0;
    }
    return [h % DIMS, (h >>> 31) ? 1 : -1];
  }
  function embed(text) {
    var v = new Array(DIMS), i;
    for (i = 0; i < DIMS; i++) v[i] = 0;
    tokensOf(text).forEach(function (word) {
      var ss = slotSign("w:" + word);
      v[ss[0]] += ss[1];
      var padded = "^" + word + "$";
      for (var j = 0; j < Math.max(1, padded.length - ORDER + 1); j++) {
        var g = slotSign("g:" + padded.substr(j, ORDER));
        v[g[0]] += g[1] * NGRAM_WEIGHT;
      }
    });
    var n = 0;
    for (i = 0; i < DIMS; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    if (n) for (i = 0; i < DIMS; i++) v[i] /= n;
    return v;
  }
  function cosine(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  // cache.research_reason: the word list plus the three shapes that make a
  // question a current one without using any of its words.
  function needsFresh(q) {
    var text = String(q).toLowerCase();
    var toks = text.replace(/[^\w\s]/g, " ").split(/\s+/);
    for (var i = 0; i < toks.length; i++) {
      if (toks[i] && VOLATILE.indexOf(toks[i]) > -1) return true;
    }
    var thisYear = new Date().getFullYear(), m, re = /\b(20\d\d)\b/g;
    while ((m = re.exec(text))) { if (Number(m[1]) >= thisYear - 1) return true; }
    if (/\bwho(?:'s|s)?\b.{0,20}\b(is|are|was|runs|leads|owns|heads|won|makes)\b/.test(text)) return true;
    return /\bhow (?:many|much)\b/.test(text);
  }

  // cache.comparable: "" if these may share an episode, else why not. The
  // reason is the point - the panel prints it, so a refusal is legible.
  function comparable(asked, stored) {
    if (numbersOf(asked) !== numbersOf(stored)) return "different numbers";
    if (needsFresh(asked) !== needsFresh(stored)) {
      return "one needs today's facts, the other does not";
    }
    var a = tokensOf(normalize(asked)), b = tokensOf(normalize(stored));
    if (!a.length || !b.length) return "nothing left after normalising";
    var both = {}, all = {}, n = 0;
    b.forEach(function (t) { both[t] = 1; });
    a.concat(b).forEach(function (t) { all[t] = 1; });
    a.forEach(function (t) { if (both[t]) n++; });
    var overlap = n / Object.keys(all).length;
    if (overlap < NEAR.overlap) {
      return "only " + overlap.toFixed(2) + " of the words in common";
    }
    return "";
  }

  // cache.best_match, over the rows already in the browser store.
  function bestMatch(query, minutes) {
    var want = embed(normalize(query)), best = null;
    rows("scripts").forEach(function (r) {
      if (!r.vector || r.minutes !== minutes || r.expires <= now()) return;
      var score = cosine(want, r.vector);
      if (score < NEAR.threshold || (best && score <= best.score)) return;
      if (comparable(query, r.query)) return;
      best = { id: r.id, query: r.query, score: score };
    });
    return best;
  }

  // topics.tags_for_text: the same keyword pass the server uses, so an episode
  // someone typed is categorised the same way a bank tile is.
  function tagsForText(text) {
    var low = " " + String(text).toLowerCase().replace(/[^\w\s]/g, " ") + " ";
    var out = [];
    Object.keys(TAG_WORDS).forEach(function (tag) {
      for (var i = 0; i < TAG_WORDS[tag].length; i++) {
        if (low.indexOf(" " + TAG_WORDS[tag][i]) !== -1) { out.push(tag); return; }
      }
    });
    return out;
  }

  // The real pipeline gets the follow-up from the model's trailing <<NEXT:>>
  // line and stores it beside the script. There is no model here, so this is a
  // deterministic stand-in keyed on the episode's own category - enough for
  // Go Deeper to be driven by what was actually listened to, which is the
  // behaviour being shown, rather than by a fixture.
  var FOLLOW = {
    sports: "what that changes for next season",
    business: "who actually profits from it",
    money: "what it does to prices",
    tech: "how it is actually built",
    science: "what the evidence still cannot settle",
    culture: "why it caught on when it did",
    health: "what the research does not claim",
    world: "who it leaves out"
  };
  // Cut on a word, never mid-word: these become card titles, and
  // "...does to interes" is the kind of thing nobody ships on purpose.
  function shorten(text, words) {
    var parts = String(text).split(/\s+/).filter(Boolean).slice(0, words || 5);
    return parts.join(" ").replace(/[,.;:\u2014-]+$/, "");
  }
  function followUp(topicId, text) {
    var t = BY_ID[topicId];
    var tags = t ? (t.tags || []) : tagsForText(text || "");
    var lead = FOLLOW[tags[0]] || "What happens next";
    var subject = t
      // Six words carries a whole bank title ("New College Football Arms
      // Race"); a typed question rarely needs more before it reads as a topic.
      ? shorten(t.title.replace(/^(The|A|An|How|Why|What|Inside)\s+/i, ""), 6)
      : shorten(String(text || "").replace(/^(how|why|what|who|when|tell me about)\s+/i, ""), 5);
    if (!subject) return lead.charAt(0).toUpperCase() + lead.slice(1);
    return lead.charAt(0).toUpperCase() + lead.slice(1) + " \u2014 " + subject.toLowerCase();
  }

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
  // `seed` is the intro's chosen interests, entering flat and before decay
  // exactly as topics.taste does - a starting position that real listening
  // overtakes rather than a rule it has to fight.
  function taste(uid, seed) {
    var s = {}, t = now(), HALF = 14 * 86400;
    (seed || []).forEach(function (g) {
      if (TAG_LABELS[g]) s[g] = INTEREST_WEIGHT;
    });
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
    var profile = taste(UID, myPrefs().interests), mine = playedIds(UID),
        used = {}, out = {};
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
      // Top four categories by episodes actually listened to, not by decayed
      // taste weight: the profile is answering "what do you listen to", and a
      // count is the honest answer to that. Ties break on taste so the order
      // is stable rather than arbitrary.
      subjects: categories(4),
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

  // Every episode this listener actually heard, categorised, most-played first.
  // Free-text searches are categorised with the same keyword pass the server
  // uses, so "why the Fed keeps rates high" counts as money like a tile would.
  function categories(n) {
    var counts = {}, t = taste(UID);
    behavioural(UID).forEach(function (e) {
      if (e.kind !== "play" && e.kind !== "complete") return;
      var tags = e.tags ? String(e.tags).split(",").filter(Boolean) : [];
      if (!tags.length) tags = tagsForText(e.text || "");
      tags.forEach(function (g) { counts[g] = (counts[g] || 0) + 1; });
    });
    return Object.keys(counts)
      .sort(function (a, b) {
        return (counts[b] - counts[a]) || ((t[b] || 0) - (t[a] || 0)) || a.localeCompare(b);
      })
      .slice(0, n || 4);
  }

  // Go Deeper: the follow-up each recently heard episode left behind.
  function threads() {
    var seen = {}, out = [];
    // Anything actually heard, newest first - a play counts, not only a
    // completion, so the section moves as soon as you listen to something.
    behavioural(UID)
      .filter(function (e) { return e.kind === "play" || e.kind === "complete"; })
      .sort(function (a, b) { return b.at - a.at; })
      .forEach(function (e) {
        var thread = e.thread || followUp(e.topic_id, e.text);
        if (!thread || seen[thread]) return;
        seen[thread] = 1;
        var t = BY_ID[e.topic_id];
        out.push({
          thread: thread, title: thread,
          from_title: t ? t.title : (e.text || "an episode"), at: e.at
        });
      });
    return out.slice(0, 8);
  }

  // ------------------------------------------------- preferences and recap
  // One row per listener, and - like the server - only read back for one with
  // an account. An anonymous listener's answers live in their own browser and
  // arrive as a hint on the request, which is what `hint` below is.
  function myPrefs() {
    var row = rows("prefs").filter(function (r) { return r.id === UID; })[0];
    return {
      interests: row && row.interests ? String(row.interests).split(",").filter(Boolean) : [],
      language: (row && row.language) || "en",
      weekly_recap: row ? row.weekly_recap !== 0 : true,
      recap_week: (row && row.recap_week) || "",
      intro_done: !!(row && row.intro_done)
    };
  }

  function hintedInterests(qs) {
    if (EMAIL) return myPrefs().interests;
    return String(qs.get("interests") || "").split(",")
      .filter(function (g) { return TAG_LABELS[g]; })
      .slice(0, MAX_INTERESTS);
  }

  // The Sunday that started the week `t` falls in, in UTC - preferences.week_start.
  function weekStart(t) {
    var d = new Date((t || now()) * 1000);
    d.setUTCDate(d.getUTCDate() - d.getUTCDay());
    return d.toISOString().slice(0, 10);
  }

  function recapBody() {
    var since = now() - 7 * 86400;
    var week = behavioural(UID).filter(function (e) { return e.at >= since; });
    var counts = {};
    week.forEach(function (e) {
      var w = WEIGHT[e.kind] || 0;
      var tags = e.tags ? String(e.tags).split(",").filter(Boolean) : tagsForText(e.text || "");
      tags.forEach(function (g) { counts[g] = (counts[g] || 0) + w; });
    });
    var subjects = Object.keys(counts).filter(function (g) { return counts[g] > 0; })
      .sort(function (a, b) { return counts[b] - counts[a]; }).slice(0, 3);
    var played = week.filter(function (e) { return e.kind === "play" || e.kind === "complete"; }).length;
    var finished = week.filter(function (e) { return e.kind === "complete"; }).length;
    var searched = week.filter(function (e) { return e.kind === "search"; }).length;
    var prefs = myPrefs();
    var body = {
      week: weekStart(), played: played, finished: finished, searched: searched,
      subjects: subjects,
      subject_labels: subjects.map(function (g) { return TAG_LABELS[g]; }),
      minutes: 5, title: "Your week in FAM", subtitle: "", query: "",
      empty: true, reason: "",
      due: prefs.weekly_recap && prefs.recap_week !== weekStart(),
      enabled: prefs.weekly_recap
    };
    if (!played && !searched) {
      body.reason = "Nothing to recap yet \u2014 this fills in once you have "
        + "listened to something this week.";
      return body;
    }
    if (!subjects.length) {
      body.reason = "You listened this week, but not to anything we could group "
        + "into a subject \u2014 so there is nothing to recap.";
      return body;
    }
    var labels = subjects.map(function (g) { return TAG_LABELS[g].toLowerCase(); });
    var joined = labels.length === 1 ? labels[0]
      : labels.slice(0, -1).join(", ") + " and " + labels[labels.length - 1];
    body.empty = false;
    body.query = "what happened this week in " + joined;
    body.subtitle = (finished ? finished + " finished \u00b7 " : "")
      + subjects.map(function (g) { return TAG_LABELS[g]; }).join(", ");
    return body;
  }

  // topics.rank_next_up: the feed's own signals over a profile seeded with the
  // episode that just ended, never a second recommender.
  function nextUpBody(topicId, text, seed) {
    var profile = taste(UID, seed);
    var heard = BY_ID[topicId] ? (BY_ID[topicId].tags || []) : tagsForText(text || "");
    heard.forEach(function (g) { profile[g] = (profile[g] || 0) + 3.0; });
    var mine = playedIds(UID), picks = [], taken = {};
    if (topicId) taken[topicId] = 1;
    function add(list, skipPlayed) {
      for (var i = 0; i < list.length && picks.length < 4; i++) {
        var t = list[i];
        if (!t || taken[t.id]) continue;
        if (skipPlayed && mine[t.id]) continue;
        taken[t.id] = 1; picks.push(t);
      }
    }
    var scored = BANK.map(function (t) {
      var sc = 0; (t.tags || []).forEach(function (g) { sc += (profile[g] || 0); });
      return { t: t, s: sc / Math.sqrt((t.tags || []).length || 1) };
    }).filter(function (x) { return x.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .map(function (x) { return x.t; });
    add(scored, true);
    add(BANK, true);
    add(BANK, false);   // four tiles is the layout; two empty squares is not
    if (UID) {
      picks.forEach(function (t) {
        addDoc("events", {
          user_id: UID, kind: "impression", topic_id: t.id, text: "",
          tags: (t.tags || []).join(","), at: now(), thread: "",
          section: "next_up", algo: ALGO
        });
      });
    }
    return { topics: picks, algo: ALGO };
  }

  // topics.rank_might_like: adjacent to a taste rather than inside it. The
  // strongest tag is suppressed on purpose - that is the whole signal.
  function exploreNewBody(seed) {
    var profile = taste(UID, seed), mine = playedIds(UID);
    var keys = Object.keys(profile);
    var top = keys.sort(function (a, b) { return profile[b] - profile[a]; })[0];
    var muted = {};
    keys.forEach(function (g) { if (g !== top) muted[g] = profile[g]; });
    var picks = [], taken = {};
    function add(list) {
      for (var i = 0; i < list.length && picks.length < 6; i++) {
        var t = list[i];
        if (!t || taken[t.id] || mine[t.id]) continue;
        taken[t.id] = 1; picks.push(t);
      }
    }
    if (keys.length) {
      add(BANK.map(function (t) {
        var sc = 0; (t.tags || []).forEach(function (g) { sc += (muted[g] || 0); });
        if ((t.tags || []).some(function (g) { return profile[g] === undefined; })) sc *= 1.4;
        return { t: t, s: sc };
      }).filter(function (x) { return x.s > 0; })
        .sort(function (a, b) { return b.s - a.s; })
        .map(function (x) { return x.t; }));
    }
    add(BANK);
    return {
      topics: picks, personalised: keys.length > 0,
      reason: keys.length
        ? "Next to what you already listen to, rather than more of it."
        : "A spread across the whole bank, until there is something to be next to.",
      algo: ALGO
    };
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
  function json(body, status, extraHeaders) {
    var headers = { "Content-Type": "application/json" };
    Object.keys(extraHeaders || {}).forEach(function (k) {
      headers[k] = extraHeaders[k];
    });
    return Promise.resolve(new Response(JSON.stringify(body), {
      status: status || 200, headers: headers
    }));
  }

  // The share wording, from sharing.py at build time, so the preview and the
  // server cannot show different copy for the same button.
  var SHARE_TEMPLATES = __SHARE_TEMPLATES__;
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

  var ACCOUNT_REQUIRED = "You need an account for this. Signing up keeps the "
    + "listening you have already done \u2014 it does not start you over.";

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

    // ---- preferences, the recap, and what plays next
    if (path === "/api/preferences" && method === "GET") {
      var stored = myPrefs();
      return json({
        interests_available: Object.keys(TAG_LABELS).map(function (g) {
          return { id: g, label: TAG_LABELS[g] };
        }),
        languages: LANGUAGES, max_interests: MAX_INTERESTS,
        language_active: false,
        account: !!EMAIL, saved: !!EMAIL,
        account_required: ACCOUNT_REQUIRED,
        interests: EMAIL ? stored.interests : [],
        language: EMAIL ? stored.language : "en",
        weekly_recap: stored.weekly_recap, recap_week: stored.recap_week,
        intro_done: EMAIL ? stored.intro_done : false
      });
    }
    if (path === "/api/preferences" && method === "POST") {
      if (!EMAIL) return json({ error: ACCOUNT_REQUIRED }, 401);
      var was = myPrefs();
      var chosen = (body.interests !== undefined && body.interests !== null)
        ? body.interests.filter(function (g) { return TAG_LABELS[g]; })
        : was.interests;
      if (chosen.length > MAX_INTERESTS) {
        return json({ error: "Choose at most " + MAX_INTERESTS + " interests." }, 400);
      }
      return put("prefs", UID, {
        interests: chosen.join(","),
        language: body.language !== undefined && body.language !== null
          ? body.language : was.language,
        weekly_recap: body.weekly_recap !== undefined && body.weekly_recap !== null
          ? (body.weekly_recap ? 1 : 0) : (was.weekly_recap ? 1 : 0),
        recap_week: was.recap_week,
        intro_done: body.intro_done !== undefined && body.intro_done !== null
          ? (body.intro_done ? 1 : 0) : (was.intro_done ? 1 : 0),
        updated: now()
      }).then(function () { paint(); return json(myPrefs()); });
    }
    if (path === "/api/recap" && method === "GET") {
      if (!EMAIL) return json({ error: ACCOUNT_REQUIRED }, 401);
      return json(recapBody());
    }
    if (path === "/api/recap/seen") {
      if (!EMAIL) return json({ error: ACCOUNT_REQUIRED }, 401);
      var before = myPrefs();
      return put("prefs", UID, {
        interests: before.interests.join(","), language: before.language,
        weekly_recap: before.weekly_recap ? 1 : 0, recap_week: weekStart(),
        intro_done: before.intro_done ? 1 : 0, updated: now()
      }).then(function () { paint(); return json({ ok: true }); });
    }
    if (path === "/api/nextup") {
      return json(nextUpBody(qs.get("topic_id") || "", qs.get("q") || "",
                             hintedInterests(qs)));
    }
    if (path === "/api/explorenew") {
      return json(exploreNewBody(hintedInterests(qs)));
    }

    // Mixes are kept for you, so they need an account - see ACCOUNT_REQUIRED
    // in app.py. Mirrored here rather than left open, because a preview that
    // is more permissive than the server hides exactly this decision.
    if (path.indexOf("/api/mixes") === 0 && !EMAIL) {
      return json({ error: ACCOUNT_REQUIRED }, 401);
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

    // ---- save for later, downloads and sharing ----
    // Held in the fixture object rather than in the artifact db: the shelf is
    // per-listener and this build's db is shared by everyone looking at the
    // link, so persisting it would show one viewer another viewer's saves.
    // The flow is what this preview is for; the storage has its own tests.
    if (path === "/api/saved" && method === "POST") {
      var shelf = FIXTURES["/api/saved"];
      var already = shelf.items.filter(function (i) {
        return i.query === body.query && i.minutes === body.minutes; })[0];
      var item = already || {
        id: "sav_" + rid(), folder_id: body.folder_id || "",
        query: body.query, minutes: body.minutes || 3,
        title: body.title || body.query, source: body.source || "",
        created: now(), downloaded: false, bytes: 0, downloaded_at: 0,
        last_played: 0,
        estimated_bytes: (body.minutes || 3) * 60 * 22050 * 2
      };
      if (!already) shelf.items.unshift(item);
      return json({ ok: true, item: item, downloads: shelf.downloads });
    }
    if (path === "/api/saved/folders" && method === "POST") {
      var folder = { id: "fld_" + rid(), name: body.name, created: now(), items: 0 };
      FIXTURES["/api/saved"].folders.push(folder);
      return json({ ok: true, folder: folder });
    }
    if (path.indexOf("/api/saved/") === 0) {
      var bits = path.split("/");
      var sid = bits[3], verb = bits[4] || "";
      var shelf2 = FIXTURES["/api/saved"];
      var found = shelf2.items.filter(function (i) { return i.id === sid; })[0];
      if (verb === "download" && method === "POST") {
        if (!found) return json({ error: "No such saved episode." }, 404);
        if (shelf2.downloads.remaining <= 0) {
          return json({ error: "You are holding " + shelf2.downloads.used
            + " downloaded episodes, which is all your plan keeps offline. "
            + "Remove one to make room." }, 409, {
              "X-FAM-Downloads": JSON.stringify({
                candidates: shelf2.items.filter(function (i) { return i.downloaded; }),
                status: shelf2.downloads })
            });
        }
        found.downloaded = true; found.bytes = found.estimated_bytes;
        shelf2.downloads.used += 1; shelf2.downloads.remaining -= 1;
        return json({ ok: true, item: found, downloads: shelf2.downloads,
                      stream: "/api/audio?q=" + encodeURIComponent(found.query)
                              + "&minutes=" + found.minutes + "&fmt=pcm" });
      }
      if (verb === "download" && method === "DELETE") {
        if (found && found.downloaded) {
          found.downloaded = false; found.bytes = 0;
          shelf2.downloads.used -= 1; shelf2.downloads.remaining += 1;
        }
        return json({ ok: true, downloads: shelf2.downloads });
      }
      if (verb) return json({ ok: true, item: found || null });
      if (method === "DELETE") {
        var at = shelf2.items.indexOf(found);
        if (at >= 0) {
          if (found.downloaded) {
            shelf2.downloads.used -= 1; shelf2.downloads.remaining += 1;
          }
          shelf2.items.splice(at, 1);
        }
        return json({ ok: true });
      }
    }
    if (path === "/api/share" && method === "POST") {
      var link = "/s/preview";
      var made = {};
      SHARE_TEMPLATES.forEach(function (t) {
        made[t.key] = {
          target: t.key, label: t.label, kind: t.kind,
          needs_image: t.needs_image, url: link, subject: "",
          text: t.text.replace("{title}", body.title || "A FAM episode")
                      .replace("{question}", body.query || "")
                      .replace("{minutes}", body.minutes || 3)
                      .replace("{url}", link)
        };
      });
      return json({ share: { id: "preview" }, url: link, public: false,
                    card: "/api/share/card?share=preview", targets: made });
    }

    if (FIXTURES[path]) return json(FIXTURES[path]);
    return json({ error: "Not available in this build." }, 404);
  }

  // Every lookup the cache has resolved, newest first. The panel reads this:
  // a hit rate is a number, and this is the thing the number is made of.
  var LOG = [];
  var EPISODE_COST = 0.0096;   // measured, see PROBLEMS.md 68

  // The script cache on the generation path, in the order the pipeline does
  // it: exact key, then - only on a miss - the near-match scan, then write.
  function touchScript(query, minutes) {
    var key = keyFor(query, minutes);
    return getOne("scripts", key).then(function (hit) {
      var live = hit && hit.expires > now();
      if (live) return bump(key, hit, "exact", query, null);

      var near = bestMatch(query, minutes);
      if (near) {
        return getOne("scripts", near.id).then(function (row) {
          if (!row) return write(key, query, minutes, near);
          return bump(near.id, row, "near", query, near);
        });
      }
      return write(key, query, minutes, null);
    });
  }
  function bump(id, row, how, asked, near) {
    note(how, asked, near, row.query);
    var d = {};
    for (var k in row) d[k] = row[k];
    d.hits = (row.hits || 0) + 1;
    return put("scripts", id, d).then(function () { paint(); return true; });
  }
  function write(key, query, minutes, near) {
    note("miss", query, near, "");
    return put("scripts", key, {
      query: query, minutes: minutes,
      sentences: "(no model call in this build)",
      created: now(), expires: now() + 86400, hits: 1,
      thread: "what that changes next",
      // The two columns the near-match cache added. The vector is written
      // here, on the write, for the reason the whole design turns on: doing
      // it on the read would put work in front of the first word.
      bucket: "m" + minutes + ":hashing:" + DIMS,
      vector: embed(normalize(query))
    }).then(function () { paint(); return false; });
  }
  // Why a lookup landed where it did. A near miss records the closest thing
  // it found and the guard that refused it, because "no match" with no reason
  // is the kind of report this project keeps getting caught by.
  function note(how, asked, near, matched) {
    var why = "";
    if (how === "miss") {
      var closest = null;
      rows("scripts").forEach(function (r) {
        if (!r.vector || r.expires <= now()) return;
        var sc = cosine(embed(normalize(asked)), r.vector);
        if (!closest || sc > closest.score) closest = { query: r.query, score: sc, minutes: r.minutes };
      });
      if (closest && closest.score >= 0.3) {
        // The guard that refused it, or - if no guard did - the fact that it
        // did not score high enough. Those are the only two ways to miss.
        why = comparable(asked, closest.query) ||
          "scored " + closest.score.toFixed(3) + ", under the " + NEAR.threshold + " threshold";
        near = { query: closest.query, score: closest.score };
      } else if (closest) {
        // Naming a "closest" that scored near zero is worse than saying
        // nothing: it reads as a near miss when the cache simply held nothing
        // on the subject.
        why = "nothing in the cache is about this";
      } else {
        why = "nothing in the cache yet";
      }
    }
    LOG.unshift({
      how: how, asked: asked, matched: matched,
      score: near ? near.score : 0, near: near ? near.query : "", why: why,
      at: now()
    });
    LOG = LOG.slice(0, 40);
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
          // Free text is categorised on the way in, exactly as app.py does it,
          // so the profile can count it.
          tags: (tid && BY_ID[tid] ? (BY_ID[tid].tags || []) : tagsForText(q)).join(","),
          at: now(), thread: followUp(tid, q), section: "", algo: ""
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

  // ------------------------------------------------------------- inspector
  // The panel beside the app. Its whole job is to make the storage visible:
  // which collection each tap lands in, what the row looks like, and the two
  // counters that carry the design - every event row against the subset the
  // ranking is allowed to read, and the cache hit rate that decides the bill.
  var COLTAB = "events", panel = null, OPEN = {};
  var SCHEMA = {
    events:   ["kind", "topic_id", "tags", "section", "algo", "at"],
    scripts:  ["id", "query", "minutes", "hits", "bucket", "expires"],
    people:   ["id", "name", "handle", "joined", "last_seen"],
    sessions: ["id", "user_id", "expires", "last_used"],
    accounts: ["id", "email", "created", "last_login"],
    mixes:    ["name", "user_id", "items", "public"],
    echoes:   ["user_id", "query", "minutes", "at"]
  };
  var FILE = {
    events: "myfam.db", scripts: "scripts.db", people: "social.db",
    sessions: "accounts.db", accounts: "accounts.db", mixes: "mixes.db",
    echoes: "social.db"
  };
  var fresh = {}, lastSeenIds = {};

  function esc(v) {
    return String(v).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function fmt(col, r, f) {
    var v = f === "id" ? r.id : r[f];
    if (v === undefined || v === "" || v === null) return '<i>-</i>';
    // A 256-float vector is data, but printing it is not showing it. The
    // shape is what tells you the row is embedded and which space it is in.
    if (Array.isArray(v)) return '<em class="vec">' + v.length + ' floats</em>';
    if (["at", "created", "joined", "last_seen", "last_login", "expires", "last_used"].indexOf(f) > -1) {
      return new Date(v * 1000).toLocaleTimeString();
    }
    if (col === "events" && f === "kind") {
      return '<em class="' + (v === "impression" ? "imp" : "beh") + '">' + esc(v) + '</em>';
    }
    var t = String(v);
    return esc(t.length > 30 ? t.slice(0, 29) + "\u2026" : t);
  }

  function paint() {
    if (!panel) return;
    // Mark rows that appeared since the last paint, so a write is visible.
    COLS.forEach(function (c) {
      var ids = {};
      rows(c).forEach(function (r) {
        ids[r.id] = 1;
        if (lastSeenIds[c] && !lastSeenIds[c][r.id]) fresh[r.id] = Date.now();
      });
      lastSeenIds[c] = ids;
    });

    var all = rows("events");
    var imps = all.filter(function (e) { return e.kind === "impression"; }).length;
    var sc = rows("scripts");
    var plays = sc.reduce(function (a, x) { return a + (x.hits || 0); }, 0);
    var hits = Math.max(0, plays - sc.length);
    var rate = plays ? Math.round(hits / plays * 100) : 0;

    panel.querySelector(".fd-state").textContent =
      db ? "connected \u00b7 persistent and shared" : "in memory \u00b7 nothing is kept";
    panel.querySelector(".fd-dot").className = "fd-dot " + (db ? "on" : "off");

    panel.querySelector(".fd-metrics").innerHTML =
      metric("rows in events", all.length, (all.length - imps) + " behavioural + " + imps + " impressions", "") +
      metric("the ranking reads", all.length - imps, "impressions excluded, so the feed cannot train on itself", "good") +
      metric("cache hit rate", rate + "%", hits + " of " + plays + " plays reused a script", rate >= 50 ? "good" : "warn") +
      metric("model spend", "$" + (sc.length * 0.0096).toFixed(3), "$" + (hits * 0.0096).toFixed(3) + " saved by the cache", "");

    panel.querySelector(".fd-tabs").innerHTML =
      '<button class="fd-tab live' + (COLTAB === "@log" ? " on" : "") + '" data-col="@log">' +
      'cache activity <u>' + LOG.length + '</u></button>' +
      COLS.map(function (c) {
        return '<button class="fd-tab' + (COLTAB === c ? " on" : "") + '" data-col="' + c + '">' +
          c + ' <u>' + rows(c).length + '</u></button>';
      }).join("");

    if (COLTAB === "@log") { panel.querySelector(".fd-table").innerHTML = activity(); return; }

    var fields = SCHEMA[COLTAB];
    var list = rows(COLTAB).slice().sort(function (a, b) {
      return (b.at || b.created || b.last_seen || b.joined || 0) -
             (a.at || a.created || a.last_seen || a.joined || 0);
    }).slice(0, 60);
    panel.querySelector(".fd-table").innerHTML =
      '<thead><tr>' + fields.map(function (f) { return '<th>' + f + '</th>'; }).join("") + '</tr></thead>' +
      '<tbody>' + (list.length ? list.map(function (r) {
        var isNew = fresh[r.id] && Date.now() - fresh[r.id] < 6000;
        var open = OPEN[r.id];
        return '<tr class="fd-row' + (isNew ? ' new' : '') + (open ? ' open' : '') +
          '" data-row="' + esc(r.id) + '">' + fields.map(function (f) {
            return '<td>' + fmt(COLTAB, r, f) + '</td>';
          }).join("") + '</tr>' + (open ? document_(r, fields.length) : '');
      }).join("") :
        '<tr><td colspan="' + fields.length + '" class="fd-empty">No rows yet in <b>' +
        COLTAB + '</b> &mdash; it lives in <b>' + FILE[COLTAB] + '</b>.</td></tr>') + '</tbody>';
  }

  // The whole document, for a row someone clicked. The table above shows the
  // handful of fields that fit; this is everything the store actually holds,
  // which is the difference between a summary of the data and the data.
  function document_(r, span) {
    var keys = Object.keys(r).sort();
    return '<tr class="fd-doc"><td colspan="' + span + '"><dl>' + keys.map(function (k) {
      var v = r[k];
      if (Array.isArray(v)) {
        if (k === "vector") {
          // The first six coordinates of a 256-slot hashed vector are almost
          // always zero, so printing them showed a column of 0.000 and said
          // nothing. The occupied slots and the unit norm are what actually
          // describe it.
          var nz = v.filter(function (n) { return n !== 0; });
          var mag = Math.sqrt(v.reduce(function (a, n) { return a + n * n; }, 0));
          v = v.length + " floats \u00b7 " + nz.length + " non-zero \u00b7 |v| = " +
            mag.toFixed(3) + " \u2014 " +
            nz.slice(0, 5).map(function (n) { return n.toFixed(3); }).join(", ") + ", \u2026";
        } else {
          v = "[" + v.join(", ") + "]";
        }
      } else if (v && typeof v === "object") {
        v = JSON.stringify(v);
      }
      return '<dt>' + esc(k) + '</dt><dd>' + esc(String(v)) + '</dd>';
    }).join("") + '</dl></td></tr>';
  }

  // Every cache lookup, and what decided it. This is the near-match cache
  // working rather than being described: an exact hit, a near hit with the
  // question it actually matched and the cosine, or a miss with the guard
  // that refused the closest thing there was.
  function activity() {
    if (!LOG.length) {
      return '<tbody><tr><td class="fd-empty">Nothing looked up yet. Search for ' +
        'something, then search for the same thing in different words.</td></tr></tbody>';
    }
    var near = LOG.filter(function (l) { return l.how === "near"; }).length;
    var hits = LOG.filter(function (l) { return l.how !== "miss"; }).length;
    return '<tbody>' +
      '<tr><td class="fd-sum">' + hits + ' of ' + LOG.length + ' lookups reused a script' +
      (near ? ', ' + near + ' of them only because of near matching &mdash; $' +
        (near * EPISODE_COST).toFixed(4) + ' that exact keys would have spent' : '') +
      '</td></tr>' +
      LOG.map(function (l) {
        var tag = l.how === "exact" ? "EXACT HIT" : l.how === "near" ? "NEAR HIT" : "MISS";
        var detail = l.how === "near"
          ? 'matched <b>' + esc(l.matched) + '</b> at ' + l.score.toFixed(3) +
            ' &mdash; every guard passed'
          : l.how === "exact"
            ? 'same normalised question, so the same key'
            : (l.near
                ? 'closest was <b>' + esc(l.near) + '</b> &mdash; ' + esc(l.why)
                : esc(l.why));
        return '<tr class="fd-log ' + l.how + '"><td>' +
          '<span class="fd-tag">' + tag + '</span> ' + esc(l.asked) +
          '<em>' + detail + '</em></td></tr>';
      }).join("") + '</tbody>';
  }
  function metric(label, value, note, tone) {
    return '<div class="fd-metric ' + (tone || "") + '"><span>' + label + '</span><b>' +
      value + '</b><em>' + note + '</em></div>';
  }

  function makePanel() {
    panel = document.getElementById("famDb");
    if (!panel) return;

    // Gather the app's own top-level nodes into one grid cell. Done here, not
    // in the markup: a wrapper element inserted around the body was closed
    // early by the app's own HTML, which left every section as its own grid
    // child. Scripts are left where they are - they do not render, so they
    // create no cell, and moving one is a needless risk.
    var stage = document.createElement("div");
    stage.id = "famStage";
    document.body.insertBefore(stage, panel);
    Array.prototype.slice.call(document.body.children).forEach(function (el) {
      if (el === stage || el === panel || el.tagName === "SCRIPT") return;
      stage.appendChild(el);
    });
    panel.innerHTML =
      '<div class="fd-head"><h2>Live database</h2>' +
      '<p>Every tap on the phone writes here. New rows are green; click any row ' +
      'for the whole document. <b>cache activity</b> shows each search resolving.</p>' +
      '<div class="fd-status"><i class="fd-dot"></i><span class="fd-state"></span></div></div>' +
      '<div class="fd-metrics"></div>' +
      '<div class="fd-tabs"></div>' +
      '<div class="fd-scroll"><table class="fd-table"></table></div>' +
      '<div class="fd-foot"><button class="fd-btn" id="fdReset">Clear all data</button>' +
      '<span class="fd-note">Session token is in localStorage here; the server uses an HttpOnly cookie.</span></div>';
    panel.querySelector(".fd-tabs").addEventListener("click", function (e) {
      var b = e.target.closest(".fd-tab"); if (!b) return;
      COLTAB = b.getAttribute("data-col"); paint();
    });
    panel.querySelector(".fd-scroll").addEventListener("click", function (e) {
      var tr = e.target.closest(".fd-row"); if (!tr) return;
      var id = tr.getAttribute("data-row");
      if (OPEN[id]) delete OPEN[id]; else OPEN[id] = 1;
      paint();
    });
    document.getElementById("fdReset").onclick = function () {
      if (!confirm("Delete every document this prototype has written?")) return;
      var jobs = [];
      COLS.forEach(function (c) { rows(c).forEach(function (r) { jobs.push(del(c, r.id)); }); });
      Promise.all(jobs).then(function () {
        try {
          localStorage.removeItem("fam_live_session");
          localStorage.removeItem("fam_resume");
        } catch (e) {}
        location.reload();
      });
    };
    paint();
    setInterval(paint, 2000);   // ages out the green highlight

    // The app's own DEMO MODE sentence tells you to add a key to .env and
    // restart. Accurate for a server, meaningless in a published page, so say
    // what is actually true of this build instead. The app's element and
    // styling are left alone - only the words change.
    setTimeout(function () {
      var el = document.getElementById("audioMode");
      if (!el) return;
      el.style.color = "var(--copper)";
      el.textContent = "PROTOTYPE \u00b7 the database on the right is real and every tap is " +
        "stored; there is no model and no speech engine here, so playback is silence of the " +
        "right length and no script is written.";
      var canned = document.getElementById("famLoadingStatus");
      if (canned) canned.textContent = "Prototype \u2014 no script is being written\u2026";
    }, 1200);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", makePanel);
  } else { makePanel(); }

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
        // Embedded on the way in, like every other write - otherwise the
        // seeded episodes are invisible to near matching and a first visit
        // could only ever miss.
        jobs.push(put("scripts", keyFor(t.query, 3), {
          query: t.query, minutes: 3, sentences: "(seeded — no model call)",
          created: now() - 3600 * (1 + j), expires: now() + 86400,
          hits: 1 + ((i + j) % 4), thread: "what that changes next",
          bucket: "m3:hashing:" + DIMS, vector: embed(normalize(t.query))
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


STAGE = """
<style>
/* The app is untouched; it is simply put in the left column of a stage and an
   inspector added on the right. Only .toast is position:fixed in the app, and
   it is fixed to the viewport either way, so nothing here disturbs it. */
  body { background: #14111c; }
  #famStage { min-width: 0; }
  #famDb {
    display: none; min-width: 0; color: #f1eef7;
    font-family: 'Public Sans', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
  }
  @media (min-width: 1040px) {
    /* The app needs about 620px; everything past that is dead space beside a
       phone, and the table would rather have it. */
    body { display: grid; grid-template-columns: minmax(0, 640px) minmax(420px, 1fr);
           align-items: start; gap: 0; justify-content: center; }
    #famStage { justify-self: center; width: 100%; }
    #famDb { display: block; position: sticky; top: 0; height: 100vh;
             border-left: 1px solid #3a3348; background: #1c1926;
             display: flex; flex-direction: column; }
  }
  /* Below that width the phone is the whole point; the panel goes underneath
     rather than squeezing it. */
  @media (max-width: 1039px) {
    #famDb { display: block; border-top: 1px solid #3a3348; background: #1c1926; }
  }
  #famDb .fd-head { padding: 16px 18px 13px; border-bottom: 1px solid #3a3348; }
  #famDb h2 { font-family: 'Fraunces', Georgia, serif; font-size: 19px; font-weight: 600;
              margin: 0 0 4px; color: #f1eef7; letter-spacing: -.01em; }
  #famDb .fd-head p { margin: 0; font-size: 12.5px; color: #a79eba; line-height: 1.45; }
  #famDb .fd-status { display: flex; align-items: center; gap: 7px; margin-top: 9px;
                      font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #a79eba; }
  #famDb .fd-dot { width: 7px; height: 7px; border-radius: 50%; background: #7c7391; }
  #famDb .fd-dot.on { background: #6bc2a8; box-shadow: 0 0 0 3px rgba(107,194,168,.16); }
  #famDb .fd-dot.off { background: #de8fa8; }
  #famDb .fd-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
                       background: #3a3348; border-bottom: 1px solid #3a3348; }
  #famDb .fd-metric { background: #1c1926; padding: 11px 14px; }
  #famDb .fd-metric span { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                           letter-spacing: .09em; text-transform: uppercase; color: #7c7391; }
  #famDb .fd-metric b { display: block; font-family: 'Fraunces', Georgia, serif;
                        font-size: 21px; line-height: 1.15; margin: 1px 0 2px; }
  #famDb .fd-metric em { font-style: normal; display: block; font-size: 11px;
                         color: #a79eba; line-height: 1.35; }
  #famDb .fd-metric.good b { color: #6bc2a8; }
  #famDb .fd-metric.warn b { color: #e9bc63; }
  #famDb .fd-tabs { display: flex; flex-wrap: wrap; gap: 5px; padding: 11px 14px;
                    border-bottom: 1px solid #3a3348; }
  #famDb .fd-tab { background: #262233; border: 1px solid #3a3348; border-radius: 7px;
                   padding: 4px 9px; font-family: 'JetBrains Mono', monospace;
                   font-size: 10.5px; color: #a79eba; cursor: pointer; }
  #famDb .fd-tab.on { background: #302a40; color: #f1eef7; border-color: #8a6a22; }
  #famDb .fd-tab u { text-decoration: none; color: #e9bc63; font-weight: 700; }
  #famDb .fd-scroll { flex: 1; overflow: auto; }
  #famDb table { width: 100%; border-collapse: collapse;
                 font-family: 'JetBrains Mono', monospace; font-size: 10.5px; }
  #famDb th { position: sticky; top: 0; background: #262233; text-align: left;
              padding: 7px 10px; font-size: 9px; letter-spacing: .07em;
              text-transform: uppercase; color: #7c7391; font-weight: 500;
              border-bottom: 1px solid #3a3348; white-space: nowrap; }
  #famDb td { padding: 6px 10px; border-bottom: 1px solid #2a2536; color: #a79eba;
              white-space: nowrap; }
  #famDb td i { color: #5d5670; font-style: normal; }
  #famDb tr.new td { background: #17322c; }
  #famDb td em { font-style: normal; padding: 1px 5px; border-radius: 4px;
                 font-size: 9px; font-weight: 700; letter-spacing: .04em;
                 text-transform: uppercase; }
  #famDb td em.imp { background: #1e2739; color: #86a9de; }
  #famDb td em.beh { background: #17322c; color: #6bc2a8; }
  #famDb td em.vec { background: #2b2440; color: #b79ae0; }
  #famDb .fd-tab.live { border-color: #6bc2a8; color: #cfe9df; }
  #famDb .fd-row { cursor: pointer; }
  #famDb .fd-row:hover td { background: #262233; }
  #famDb .fd-row.open td { background: #262233; color: #f1eef7; }
  /* The whole document under the row it belongs to. Wraps, unlike the table
     above it, because the point is to read the values rather than scan them. */
  #famDb .fd-doc td { white-space: normal; background: #16131f; padding: 9px 12px 11px; }
  #famDb .fd-doc dl { margin: 0; display: grid; grid-template-columns: 88px 1fr;
                      gap: 3px 10px; }
  #famDb .fd-doc dt { color: #7c7391; font-size: 9.5px; letter-spacing: .05em;
                      text-transform: uppercase; padding-top: 1px; }
  #famDb .fd-doc dd { margin: 0; color: #dcd6e8; font-size: 10.5px;
                      word-break: break-word; }
  #famDb .fd-sum { white-space: normal; background: #1f1b2b; color: #dcd6e8;
                   padding: 10px 12px; font-size: 11px; line-height: 1.5; }
  #famDb .fd-log td { white-space: normal; line-height: 1.5; color: #dcd6e8;
                      padding: 9px 12px; }
  #famDb .fd-log em { display: block; margin-top: 4px; padding: 0; background: none;
                      text-transform: none; letter-spacing: 0; font-weight: 400;
                      font-size: 10px; color: #a79eba; }
  #famDb .fd-log b { color: #f1eef7; font-weight: 600; }
  #famDb .fd-tag { display: inline-block; margin-right: 6px; padding: 1px 6px;
                   border-radius: 4px; font-size: 9px; font-weight: 700;
                   letter-spacing: .05em; }
  #famDb .fd-log.exact .fd-tag { background: #17322c; color: #6bc2a8; }
  #famDb .fd-log.near .fd-tag  { background: #2b2440; color: #c4a6f0; }
  #famDb .fd-log.miss .fd-tag  { background: #3a2430; color: #de8fa8; }
  #famDb .fd-empty { white-space: normal; padding: 20px; text-align: center; color: #a79eba; }
  #famDb .fd-foot { padding: 10px 14px; border-top: 1px solid #3a3348;
                    display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  #famDb .fd-btn { background: #262233; border: 1px solid #3a3348; border-radius: 7px;
                   padding: 6px 11px; font-size: 12px; color: #a79eba; cursor: pointer;
                   font-family: inherit; }
  #famDb .fd-btn:hover { color: #f1eef7; border-color: #8a6a22; }
  #famDb .fd-note { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                    color: #7c7391; line-height: 1.4; flex: 1; min-width: 150px; }
</style>
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
    import cache  # noqa: E402
    import preferences as prefs_mod  # noqa: E402
    import sharing  # noqa: E402
    import topics  # noqa: E402
    from config import settings  # noqa: E402

    shim = (LIVE_SHIM
            .replace("__FIXTURES__", json.dumps(bp.load_fixtures()))
            .replace("__ALGO__", json.dumps(topics.ALGO_VERSION))
            .replace("__SHARE_TEMPLATES__", json.dumps([
                {"key": t.key, "label": t.label, "kind": t.kind,
                 "needs_image": t.needs_image, "text": t.template}
                for t in sharing.TARGETS]))
            .replace("__TAG_WORDS__", json.dumps(
                {k: list(v) for k, v in topics.TAG_WORDS.items()}))
            # The intro's vocabulary, from the modules that own it. Typed out
            # here it would drift, and a picker offering a facet the ranker
            # does not score is the drift that matters.
            .replace("__TAG_LABELS__", json.dumps(topics.TAG_LABELS))
            .replace("__LANGUAGES__", json.dumps([dict(l) for l in prefs_mod.LANGUAGES]))
            .replace("__MAX_INTERESTS__", json.dumps(prefs_mod.MAX_INTERESTS))
            .replace("__INTEREST_WEIGHT__", json.dumps(topics.INTEREST_WEIGHT))
            .replace("__VOLATILE__", json.dumps(sorted(cache.research_words())))
            .replace("__NEAR__", json.dumps({
                "threshold": settings.cache_vector_threshold,
                "overlap": settings.cache_vector_overlap,
            }))
            .replace("__STATIC_PATHS__", json.dumps(list(STATIC_PATHS))))

    at = html.index("<script>")
    html = html[:at] + shim + html[at:]

    # The inspector is appended; the app's own nodes are gathered into a stage
    # column by the shim at load time rather than by wrapping them here. An
    # HTML wrapper was tried first and the app's markup closed it early, so
    # body ended up with seven grid children instead of two.
    html = html.replace("</head>", STAGE + "</head>", 1)
    html = html.replace("</body>", '\n<aside id="famDb"></aside>\n</body>', 1)

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
