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

        def first_run_asks_before_it_shows_the_app():
            """The entry flow runs once, and every path through it lands in
            the app. A first-run screen with no way out is the worst bug this
            file could miss, because it is the only screen everybody sees."""
            assert page.eval_on_selector(".screen.active", "e => e.id") == "screen-welcome", \
                "a first open did not start on the welcome screen"
            assert page.query_selector("#screen-welcome .entry-skip"), \
                "there was no way past the account step"
            # Signed up rather than skipped, because the gated surfaces below
            # (mixes, the recap) are the ones with something to check, and
            # skipping is asserted above as reachable.
            page.evaluate("openAuth('signup')")
            page.wait_for_timeout(400)
            page.evaluate("showAuthForm()")
            page.fill("#authEmail", "smoke@example.com")
            page.fill("#authPassword", "a-long-enough-password")
            page.evaluate("submitAuthForm()")
            page.wait_for_selector("#screen-intro.active .intro-chip",
                                   timeout=10000, state="attached")
            chips = page.eval_on_selector_all(".intro-chip", "e => e.length")
            assert chips >= 6, f"only {chips} interests offered"
            # The cap is a disabled chip, not a message after the fact.
            for i in range(7):
                page.evaluate(f"var c=document.querySelectorAll('.intro-chip')[{i}];"
                              " if(c) c.click();")
            chosen = page.eval_on_selector_all(".intro-chip.on", "e => e.length")
            assert chosen == 6, f"the six-interest cap let {chosen} through"
            assert page.query_selector(".intro-chip.full"), \
                "the seventh chip was still selectable"
            page.evaluate("introNext()")
            page.wait_for_selector("#introPageLanguage .intro-lang",
                                   timeout=10000, state="attached")
            page.evaluate("finishIntro()")
            page.wait_for_timeout(900)
            assert page.eval_on_selector(".screen.active", "e => e.id") == "screen-myfam", \
                "finishing the intro did not land in the app"

        def the_weekly_recap_pops_on_a_new_week():
            """Fixture says this week's recap is still owed, so it fires on the
            first open after the intro - and has to be dismissable."""
            page.wait_for_selector("#recapOverlay.active", timeout=10000)
            # A tile when there is a week to recap, a sentence saying so when
            # there is not. Both are correct; an empty card is not.
            assert page.text_content("#recapBody").strip(), "the recap card was blank"
            page.evaluate("closeRecap()")
            page.wait_for_timeout(400)
            assert not page.query_selector("#recapOverlay.active"), \
                "the recap could not be dismissed"

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

        def your_fam_offers_the_recap_and_explore_new():
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(500)
            page.click("#screen-myfam .myfam-msg-btn")
            page.wait_for_timeout(600)
            tiles = page.eval_on_selector_all(".yf-tile-name", "e => e.map(x => x.textContent)")
            assert tiles == ["Weekly Recap", "Explore New"], f"saw {tiles}"
            page.evaluate("openExploreNew()")
            page.wait_for_selector("#screen-explorenew.active .xn-card",
                                   timeout=10000, state="attached")
            assert page.eval_on_selector_all(".xn-card", "e => e.length") >= 4
            assert page.text_content("#xnReason").strip(), \
                "Explore New did not say why it was showing these"
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(400)

        def whats_next_offers_four_and_counts_down():
            """The popup, driven the way an ended episode drives it. The
            countdown tile is checked for existence, not waited out - five
            seconds of real time in a smoke test buys nothing."""
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(400)
            page.evaluate("showScreen('player')")
            page.evaluate("maybeOfferNextUp('what the fed did to interest rates', '')")
            page.wait_for_selector("#nextUpOverlay.active .nextup-tile",
                                   timeout=10000)
            tiles = page.eval_on_selector_all(".nextup-tile", "e => e.length")
            assert tiles == 4, f"expected a 2x2 grid, saw {tiles} tiles"
            assert page.query_selector(".nextup-tile.lead .nextup-timer"), \
                "the first tile has no countdown"
            assert "starts in" in page.text_content("#nextUpSub").lower()
            # Tapping anything else cancels the countdown rather than racing it.
            page.evaluate("closeNextUp()")
            page.wait_for_timeout(300)
            assert not page.query_selector("#nextUpOverlay.active")
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(400)

        def ensure_account():
            """Sign up unless this browser already has an account.

            Needed because one check above clears localStorage on purpose, and
            on the live preview that is a real logout: the session token lives
            there. Mixes are account-gated, so anything below that touches them
            has to put an account back first. The email is unique per call -
            the store is durable and the same address twice is refused, exactly
            as the server refuses it.
            """
            if page.evaluate("() => AUTH && AUTH.authenticated"):
                return
            page.evaluate(
                """() => {
                    var who = "smoke-" + Math.random().toString(36).slice(2, 9)
                              + "@example.com";
                    return fetch("/api/auth/signup", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({ email: who,
                                               password: "a-long-enough-password" })
                    }).then(function(){ return refreshAuth(); });
                }"""
            )
            page.wait_for_timeout(600)

        def the_account_gate_reads_as_a_choice():
            """Skipping the account step has to look like a decision, not a
            broken screen - and it has to offer the way out of itself."""
            page.evaluate("openPlayFAM()")
            # After the load settles, not with it: loadMixes writes the same
            # element asynchronously and would paint over this.
            page.wait_for_timeout(1200)
            page.evaluate("renderMixesLocked()")
            page.wait_for_selector("#screen-playfam .locked-note", timeout=10000)
            assert page.eval_on_selector_all("#screen-playfam .locked-acts .pf-btn",
                                             "e => e.length") == 2, \
                "the gate offered no way to sign up or log in"
            text = page.text_content("#screen-playfam .locked-note").lower()
            assert "start you over" in text, \
                "the gate did not say signing up keeps what they already have"

        def the_bar_can_be_dragged_to_seek():
            """Sliding the bar is a seek, and it has to be a real one.

            Driven with the mouse rather than by calling FamAudio.seek: the
            thing under test is the gesture - pointer capture, the clamp at the
            buffered edge, the class that says the bar has been picked up -
            not the seek underneath it, which the transport already had.
            """
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(500)
            page.evaluate("startBankTopic(Object.keys(myFamTopics)[0])")
            page.wait_for_selector("#screen-player.active", timeout=15000)
            # Enough audio has to have arrived for there to be anywhere to
            # seek to: the bar clamps at what has been written.
            page.wait_for_timeout(4500)
            before = page.evaluate("() => FamAudio.position()")

            box = page.eval_on_selector(
                "#screen-player .progress-bar",
                "e => { var r = e.getBoundingClientRect();"
                " return {x: r.x, y: r.y, w: r.width}; }")
            page.mouse.move(box["x"] + 4, box["y"] + 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["w"] * 0.9, box["y"] + 2, steps=8)
            assert page.query_selector("#screen-player .progress-bar.scrubbing"), \
                "the bar did not say it had been picked up"
            page.mouse.up()
            page.wait_for_timeout(400)

            after = page.evaluate("() => FamAudio.position()")
            assert after > before + 0.5, \
                f"dragging the bar did not move playback ({before:.2f} -> {after:.2f})"
            assert not page.query_selector("#screen-player .progress-bar.scrubbing"), \
                "the bar stayed picked up after the drag ended"
            # And the two gestures still coexist: the buttons were the point of
            # "on top of", not a thing this replaced.
            page.evaluate("skipAudio(-15)")
            page.wait_for_timeout(300)
            assert page.evaluate("() => FamAudio.position()") < after, \
                "the 15-second button stopped working once the bar could be dragged"
            page.evaluate("goBack()")
            page.wait_for_timeout(400)

        def explores_bar_scrubs_without_swiping():
            """The bar in Explore seeks, and does not deal the next card.

            Explore listens for swipes on an ancestor of its bar, so without
            the guard in makeScrubbable a drag along the bar is both a seek and
            a swipe - and the episode you were aiming at is gone.
            """
            page.evaluate("openExplore()")
            page.wait_for_selector("#screen-explore.active", timeout=10000)
            # Coming back to the tab keeps the listener's place but does not
            # resume - setTab stops playback - so press play the way they
            # would, then let enough audio arrive to have somewhere to seek to.
            page.evaluate("if(!FamAudio.isActive()) reelTogglePlay();")
            page.wait_for_timeout(4500)
            was = page.text_content("#reelTitle")
            before = page.evaluate("() => FamAudio.position()")
            box = page.eval_on_selector(
                "#screen-explore .reel-progress",
                "e => { var r = e.getBoundingClientRect();"
                " return {x: r.x, y: r.y, w: r.width}; }")
            page.mouse.move(box["x"] + 3, box["y"] + 1)
            page.mouse.down()
            page.mouse.move(box["x"] + box["w"] * 0.9, box["y"] + 1, steps=8)
            assert page.query_selector("#screen-explore .reel-progress.scrubbing"), \
                "the reel bar did not say it had been picked up"
            page.mouse.up()
            page.wait_for_timeout(500)
            assert page.text_content("#reelTitle") == was, \
                "dragging the bar swiped to the next episode"
            after = page.evaluate("() => FamAudio.position()")
            assert after > before + 0.5, \
                f"dragging the reel bar did not move playback ({before:.2f} -> {after:.2f})"
            page.evaluate("openMyFamTab()")
            page.wait_for_timeout(400)

        def dailyfam():
            ensure_account()
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

        def loading_screen_on_a_search():
            """The listener must be able to tell the search was received.

            This is the check the old code could not have passed: the overlay
            was chosen by screen name and the home screen's id did not exist,
            so pressing search showed the search page again and nothing else.
            """
            page.evaluate("setTab('home')")
            page.wait_for_timeout(300)
            assert not page.evaluate(
                """() => document.getElementById("famLoading").classList.contains("active")"""
            ), "the loading screen was showing before anything was asked for"

            page.fill("#searchInput", "what happened with the fed today")
            page.evaluate("runSearch()")
            page.wait_for_timeout(400)
            assert page.evaluate(
                """() => document.getElementById("famLoading").classList.contains("active")"""
            ), "pressing search showed no loading screen"

            status = page.text_content("#famLoadingStatus") or ""
            assert status.strip(), "the loading screen said nothing about what it was doing"
            # PROBLEMS.md 55: the wait names itself. A brand animation that
            # replaced that line would be the filler problem in a nicer font.
            assert ("Writing" in status or "sources" in status
                    or "sample script" in status or "rejected" in status), (
                f"the loading screen does not say what it is waiting for: {status!r}"
            )
            page.evaluate("clearGenOverlay()")
            page.wait_for_timeout(200)
            assert not page.evaluate(
                """() => document.getElementById("famLoading").classList.contains("active")"""
            ), "the loading screen did not go away"

        def loading_screen_covers_every_surface():
            """One screen, not one per tab. Four overlays chosen by id is how
            the home screen ended up with none."""
            count = page.evaluate(
                """() => document.querySelectorAll(".fam-loading").length"""
            )
            assert count == 1, f"expected one loading screen, found {count}"
            leftovers = page.evaluate(
                """() => document.querySelectorAll(".generating").length"""
            )
            assert leftovers == 0, f"{leftovers} old per-screen overlay(s) survive"

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
        check("The first run asks, then lets you in", first_run_asks_before_it_shows_the_app)
        check("The weekly recap pops and closes", the_weekly_recap_pops_on_a_new_week)
        check("myFAM renders three rails", myfam)
        check("Go Deeper titles are not cut off", go_deeper_titles_fit)
        check("Go Deeper fills for a new listener", go_deeper_fills_for_a_new_listener)
        check("A file can be attached to a search", attachments)
        check("Searching shows the loading screen", loading_screen_on_a_search)
        check("One loading screen serves every surface", loading_screen_covers_every_surface)
        check("Your FAM offers the recap and Explore New",
              your_fam_offers_the_recap_and_explore_new)
        check("What's next offers four with a countdown",
              whats_next_offers_four_and_counts_down)
        check("The bar can be dragged to seek", the_bar_can_be_dragged_to_seek)
        check("The account gate reads as a choice", the_account_gate_reads_as_a_choice)
        check("DailyFAM lists mixes", dailyfam)
        check("picker offers a typed topic", picker)
        check("Explore plays and advances", explore)
        check("Explore's bar scrubs without swiping", explores_bar_scrubs_without_swiping)
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
