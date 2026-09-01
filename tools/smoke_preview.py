"""Drive the built preview in a real browser, with no server at all.

This is the check that catches what pytest cannot: a tab that renders nothing,
a feed that throws, a reel that will not advance, audio that never starts. All
three Explore bugs were found this way by hand; this runs it every push.

    python tools/smoke_preview.py [path-to-preview.html]
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "preview" / "fam-preview.html"

sys.path.insert(0, str(ROOT))
import topics as topics_mod  # noqa: E402  - the real bank, not a fixture copy


def main() -> int:
    from playwright.sync_api import sync_playwright

    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not target.exists():
        print(f"no preview at {target} - run python preview/build_preview.py", file=sys.stderr)
        return 1

    def launch_browser(pw):
        """Playwright's own download first; any installed Chromium after.

        Environments that ship a browser at a different version than the
        Playwright package expects are common enough that failing there would
        make this check something people skip.
        """
        candidates = [os.environ.get("PLAYWRIGHT_CHROMIUM")]
        try:
            return pw.chromium.launch()
        except Exception as first:
            for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                            "/opt/pw-browsers/chromium/chrome-linux/chrome"):
                candidates += sorted(str(p) for p in pathlib.Path("/").glob(pattern.lstrip("/")))
            for path in [c for c in candidates if c and pathlib.Path(c).exists()]:
                try:
                    return pw.chromium.launch(executable_path=path)
                except Exception:
                    continue
            raise first

    failures: list[str] = []
    with sync_playwright() as pw:
        browser = launch_browser(pw)
        page = browser.new_page(viewport={"width": 430, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("file://" + str(target.resolve()))
        page.wait_for_timeout(2500)
        page.evaluate("var s=document.getElementById('splash'); if(s)s.classList.add('hide');")

        def check(label: str, fn):
            try:
                fn()
                print(f"  ok    {label}")
            except Exception as exc:  # noqa: BLE001 - report, don't stop
                failures.append(f"{label}: {exc}")
                print(f"  FAIL  {label}: {exc}")

        def myfam():
            page.evaluate("openMyFamTab()")
            page.wait_for_selector(".feed-rail .seed-card", timeout=10000, state="attached")
            rails = page.eval_on_selector_all(".feed-section", "e => e.length")
            assert rails == 3, f"expected 3 sections, saw {rails}"

        def go_deeper_titles_fit():
            """A clipped title is invisible to every other check.

            The tiles are a fixed height, so the only thing that tells you a
            headline is being cut mid-word is looking at a phone - which is
            how it shipped once. Every title the bank can produce is measured
            here, plus a thread at the longest the prompt asks for.
            """
            page.evaluate("openMyFamTab()")
            page.wait_for_selector(".gd-card-title", timeout=10000, state="attached")
            titles = [t.title for t in topics_mod.TOPIC_BANK]
            titles.append(
                # `<<NEXT: six to twelve words>>` - a thread card shows this raw.
                "What happens to the grid operators when the subsidy expires next year"
            )
            clipped = page.evaluate(
                """(xs) => {
                    var el = document.querySelector(".gd-card-title");
                    var original = el.textContent;
                    var bad = xs.filter(function(x){
                        el.textContent = x;
                        return el.scrollHeight > el.clientHeight + 1;
                    });
                    el.textContent = original;
                    return bad;
                }""",
                titles,
            )
            assert not clipped, f"Go Deeper tile cuts these titles off: {clipped}"

        def go_deeper_fills_for_a_new_listener():
            """Four tiles even with no history - the case nobody develops in.

            Everyone testing this has threads and half-heard episodes, so the
            empty section only ever appeared for someone opening the app for
            the first time. The tiles must be real bank topics (a query to
            generate from), not placeholder text, and must not repeat what the
            rails below are already showing.
            """
            page.evaluate(
                """() => {
                    try { localStorage.clear(); } catch (e) {}
                    var real = window.fetch;
                    window.fetch = function(u, o){
                        if(String(u).indexOf("/api/godeeper") === 0){
                            return Promise.resolve({ ok: true,
                                json: function(){ return Promise.resolve({ threads: [] }); } });
                        }
                        return real(u, o);
                    };
                }"""
            )
            page.evaluate("openMyFamTab(); loadMyFamFeed()")
            page.wait_for_timeout(1800)
            cards = page.evaluate("() => goDeeperCardCache")
            assert len(cards) == 4, f"a new listener saw {len(cards)} Go Deeper tiles, not 4"
            assert all(c["kind"] == "starter" for c in cards), \
                f"expected all starters, got {[c['kind'] for c in cards]}"
            assert all(c.get("query") and c.get("topicId") for c in cards), \
                "a starter tile with no query or topic id cannot generate or be logged"
            titles = page.eval_on_selector_all(".gd-card-title", "e => e.map(x => x.textContent)")
            rails = page.eval_on_selector_all(".seed-card-title", "e => e.map(x => x.textContent)")
            repeated = sorted(set(titles) & set(rails))
            assert not repeated, f"Go Deeper repeats what the rails show: {repeated}"
            label = page.text_content(".gd-count")
            assert "left off" not in label.lower(), \
                f"told a first-run listener they left something off: {label!r}"
            page.reload()
            page.wait_for_timeout(1200)

        def attachments():
            """A file becomes a chip, and the chip becomes an id on the request.

            Also pins the two rules the feature exists under: an attachment on
            its own is a summarise request rather than an error, and it is
            cleared once used so it cannot ride along on the next question.
            """
            page.evaluate("setTab('home')")
            page.wait_for_timeout(300)
            assert page.query_selector(".attach-btn"), "no way to attach anything"
            page.set_input_files("#attachFile", {
                "name": "q3-report.txt", "mimeType": "text/plain",
                "buffer": b"Revenue fell 12 percent.",
            })
            page.wait_for_timeout(800)
            chips = page.eval_on_selector_all(".attach-chip", "e => e.length")
            assert chips == 1, f"expected one chip, saw {chips}"
            name = page.text_content(".attach-chip .nm")
            assert "q3-report" in name, f"the chip does not name the file: {name!r}"
            assert page.evaluate("() => attachedIds()"), "the chip carries no id"

            # Nothing typed: the attachment itself is the request.
            page.evaluate("runSearch()")
            page.wait_for_timeout(600)
            asked = page.evaluate("() => TOPICS['_custom'] && TOPICS['_custom'].prompt")
            assert asked and "attached" in asked.lower(), \
                f"an attachment alone did not become a request: {asked!r}"
            carried = page.evaluate("() => TOPICS['_custom'].attach")
            assert carried, "the episode was generated without the attachment"
            assert page.eval_on_selector_all(".attach-chip", "e => e.length") == 0, \
                "the attachment stayed on screen and would ride along on the next search"
            page.reload()
            page.wait_for_timeout(1200)

        def dailyfam():
            page.evaluate("openPlayFAM()")
            page.wait_for_selector(".mix-card", timeout=10000, state="attached")
            assert page.eval_on_selector_all(".mix-card", "e => e.length") >= 1

        def picker():
            page.evaluate("document.querySelectorAll('.mix-card')[0].click()")
            page.wait_for_timeout(400)
            page.evaluate("editMixTopics()")
            page.wait_for_selector("#screen-mixpicker.active .mix-topic",
                                   timeout=10000, state="attached")
            page.fill("#pickerSearch", "a topic nobody has in the bank")
            page.wait_for_timeout(300)
            assert page.query_selector(".typed-offer"), "typing offers no way to add it"

        def messages_sheet():
            # The sheet has to be leavable. A tab that cannot be left is the
            # bug this app already shipped once, on Explore.
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(600)
            page.click("#screen-myfam .myfam-msg-btn")
            page.wait_for_timeout(700)
            assert page.eval_on_selector(".screen.active", "e => e.id") == "screen-messages"
            page.click("#screen-messages .sheet-close")
            page.wait_for_timeout(700)
            assert page.eval_on_selector(".screen.active", "e => e.id") == "screen-myfam", \
                "closing messages did not return to myFAM"

        def profile():
            page.evaluate("openProfile()")
            page.wait_for_timeout(1200)
            assert page.eval_on_selector(".screen.active", "e => e.id") == "screen-profile"
            assert page.query_selector(".pf-name"), "no identity block"
            assert page.eval_on_selector_all(".pf-echo", "e => e.length") > 0, "no echoes"
            assert page.eval_on_selector_all(".pf-art b", "e => e.length") > 0, "no folders"
            assert page.query_selector(".pf-headline"), "no my-FAM-is-your-FAM headline"

        def mix_visibility():
            # Public/private has to be reachable, not buried in a menu.
            page.evaluate("openPlayFAM()")
            page.wait_for_selector(".mix-card", timeout=10000, state="attached")
            page.wait_for_timeout(400)
            page.evaluate("document.querySelectorAll('.mix-card')[0].click()")
            page.wait_for_timeout(500)
            switch = page.query_selector(".mix-switch")
            assert switch, "no public/private switch inside a mix"
            before = "on" in (switch.get_attribute("class") or "")
            page.click(".mix-vis")
            page.wait_for_timeout(900)
            after = "on" in (page.query_selector(".mix-switch").get_attribute("class") or "")
            assert after != before, "the visibility switch did not move"

        #: Every screen a listener can control playback from. Echo belongs on
        #: all of them - checking two ids by name is what let the main player
        #: ship without one.
        PLAYERS = ["screen-player", "screen-playall", "screen-explore"]

        def echo_button():
            page.evaluate("openExplore()")
            page.wait_for_timeout(2200)
            missing = page.evaluate(
                """(ids) => ids.filter(function(id){
                       var el = document.getElementById(id);
                       return !el || !el.querySelector("[data-echo]");
                   })""",
                PLAYERS,
            )
            assert not missing, f"no echo control on: {missing}"

        def echo_state_reaches_every_player():
            """One echo must light up all of them, not just the one tapped."""
            page.evaluate("setEchoed(true)")
            lit = page.evaluate(
                """() => Array.from(document.querySelectorAll("[data-echo]"))
                       .filter(function(el){ return el.classList.contains("echoed"); }).length"""
            )
            total = page.evaluate("""() => document.querySelectorAll("[data-echo]").length""")
            page.evaluate("setEchoed(false)")
            still = page.evaluate(
                """() => Array.from(document.querySelectorAll("[data-echo]"))
                       .filter(function(el){ return el.classList.contains("echoed"); }).length"""
            )
            assert total >= 3, f"expected an echo control on every player, found {total}"
            assert lit == total, f"only {lit} of {total} echo controls showed the echoed state"
            assert still == 0, f"{still} echo control(s) stayed lit after un-echoing"

        def explore():
            page.evaluate("openExplore()")
            page.wait_for_timeout(2500)
            first = page.text_content("#reelTitle")
            assert first and "Loading" not in first, f"reel never loaded ({first!r})"
            page.wait_for_timeout(2000)
            assert page.evaluate("FamAudio.position()") > 0, "audio never started"
            page.evaluate("nextReel()")
            page.wait_for_timeout(1500)
            assert page.text_content("#reelTitle") != first, "swipe did not advance"

        print(f"smoke test: {target.name}")
        check("myFAM renders three rails", myfam)
        check("Go Deeper titles are not cut off", go_deeper_titles_fit)
        check("Go Deeper fills for a new listener", go_deeper_fills_for_a_new_listener)
        check("A file can be attached to a search", attachments)
        check("DailyFAM lists mixes", dailyfam)
        check("picker offers a typed topic", picker)
        check("Explore plays and advances", explore)
        check("Messages opens and closes", messages_sheet)
        check("Profile renders identity, folders and echoes", profile)
        check("Mix visibility can be toggled", mix_visibility)
        check("Echo control is on every player", echo_button)
        check("Echo state reaches every player", echo_state_reaches_every_player)

        if errors:
            failures.append(f"page errors: {errors}")
            print(f"  FAIL  page errors: {errors}")
        browser.close()

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
