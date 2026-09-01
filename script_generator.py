"""Turn a search query into a podcast script of a requested length.

Two things matter here:

1. **Length control.** The requested minutes are converted into a word budget
   at the narrator's nominal rate. The budget is given to Claude as a hard
   constraint with a per-section breakdown, because "write a 6 minute podcast"
   alone produces wildly variable lengths.

2. **Streaming.** The script is streamed sentence by sentence so speech
   synthesis can start before the model has finished writing. This is what
   makes audio playable within a couple of seconds instead of after the whole
   episode is generated.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

from anthropic_client import build_async_client
from config import settings

log = logging.getLogger(__name__)


def load_style_examples() -> list[tuple[str, str]]:
    """Briefings from examples/ that show the model the house voice.

    Rules describe a style; examples demonstrate one, and a model matches a
    demonstration far more closely than a description. This is the most direct
    control there is over how the writing sounds.

    Files are `<minutes>-<slug>.txt`: first line the query, blank line, then the
    script. Read once at import; an empty folder changes nothing.
    """
    import pathlib

    directory = pathlib.Path(__file__).resolve().parent / "examples"
    found: list[tuple[str, str]] = []
    if not directory.is_dir():
        return found

    for path in sorted(directory.glob("*.txt")):
        try:
            head, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        except OSError:
            continue
        query, script = head.strip(), body.strip()
        if query and script:
            found.append((query, script))
        else:
            log.warning("skipping style example %s: expected a query, a blank line, then the script",
                        path.name)
    if found:
        log.info("loaded %d style example(s) from examples/", len(found))
    return found


STYLE_EXAMPLES = load_style_examples()


def style_example_block() -> str:
    """The examples, formatted for the system prompt. Empty when there are none."""
    if not STYLE_EXAMPLES:
        return ""
    parts = [
        "\nThis is the house voice. Match its rhythm, its directness and the way "
        "it opens and closes. Do not borrow its facts or its topics - only how it "
        "sounds. They show the spoken script only; you must still end with the "
        "<<NEXT: ...>> line:\n"
    ]
    for query, script in STYLE_EXAMPLES:
        words = len(script.split())
        parts.append(f'<example query="{query}" words="{words}">\n{script}\n</example>\n')
    return "\n".join(parts)

# The last few openers spoken by this server. Fed back into the prompt so a
# listener working through several episodes does not hear the same shape of
# introduction every time - the single most noticeable tell of a generated show.
_RECENT_OPENERS: list[str] = []
_RECENT_LIMIT = 8


def remember_opener(text: str) -> None:
    if not text:
        return
    _RECENT_OPENERS.append(text.strip())
    del _RECENT_OPENERS[:-_RECENT_LIMIT]

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
# The one line the model may write that is not speech: the thread it left open,
# phrased as the follow-up a listener would ask for. Stripped before synthesis
# and handed to the interface, which offers it as a one-tap "go deeper".
_NEXT_MARKER = re.compile(r"<<\s*NEXT\s*:\s*([^<>]{1,160}?)\s*>>", re.I)
# Anything that would be read aloud as punctuation noise rather than speech.
_MARKDOWN = re.compile(r"[*_`#>\[\]]|^\s*[-•]\s+", re.MULTILINE)

SYSTEM_PROMPT = """You write FAM: short spoken pieces that answer what someone asked, told as a \
story - a story made *of* the facts, not wrapped around them. Done right the \
listener never notices the shape; they just find they do not want to stop.

**Answer them.** Someone asked because they wanted to know something, and they \
must finish knowing it - well enough to say it back to a friend in their own \
words. Everything below is how that answer arrives, and none of it is worth \
anything if it does not. So: satisfied first, curious second. The curiosity \
makes them want another episode; the satisfaction makes them believe another \
is worth having.

Which means what you leave open at the end is **never the answer held back**. \
Withholding is not momentum, it is a bait and switch, and a listener feels it \
immediately. Close the question they came with completely, then let the answer \
open a different one.

Before you write, find the angle:
- **Find what they have slightly wrong.** The most interesting version of \
almost any answer is a correction - what everyone assumes, and what is \
actually true instead. Look for that first. If there is nothing to correct, \
find the part that is stranger, smaller or more specific than they expect.
- **Go under the obvious answer.** Every question has a stock reply it usually \
gets. That reply is not worth an episode. Get beneath it - to the mechanism, \
the evidence, the reason the stock reply exists at all.

The opening:
- Start inside something already in motion - a process running, a moment \
happening, a number moving. Concrete enough to picture, specific enough that \
it could not open a different episode.
- Open a small "wait, why?" with that first line, then spend the piece \
answering it. Do not state your conclusion in sentence one; you have nowhere \
to go after that. Do not delay it either.
- No scene-setting for its own sake. No "picture this", no "imagine", no "it \
was a cold morning in", no throat-clearing of any kind.
- Banned outright: "Here's what I can tell you about...", "Let's talk \
about...", "This is a fascinating topic...", "There's a lot to unpack \
here...".

How it is built:
- **Because, therefore, but - not and then.** Facts in time order are a list. \
Causation is a story. Each beat should feel like it had to follow the one \
before.
- **Tension then release.** Something is unresolved, surprising or at stake; \
everything moves toward resolving it; when it resolves, you are done.
- **One concrete anchor beats three abstractions.** A named person, an actual \
figure, a specific moment.
- **Know more than you say.** Write with the confidence of someone who has \
read far more than they are telling. Never hedge, never survey "many \
perspectives", never pad with the obvious.
- **Say "you" when the question is theirs.** Someone asking how to think, \
sleep, decide or cope is asking about their own life. Talk to them, not about \
people in general.
- **If they asked how to think or what to do, leave them something usable.** \
One thing they could actually do tomorrow beats a paragraph of principle.

Take them in rather than showing them round:
- **Speak from inside.** Assume the listener is already here. No orienting \
them, no "as you may know", no explaining why this is worth their time - \
explaining that is proof it is not.
- **Have a point of view.** Say which account is better supported, which claim \
is weak, what is actually surprising. Neutral survey is how something reads as \
generated rather than told.

How it ends, which decides whether they stay:
- **Never build an exit. Do not end. Widen.** A closed loop is a place to \
stop. Resolve the question you opened, then let the answer raise the next one, \
so the natural move is forward rather than away. Never summarise, never recap. \
They should finish inside the subject, not outside it holding a summary of it.
- **Leave exactly one thread, and never the main one.** It is second-order: \
something the answer itself raised, that they could not have known to ask when \
they started. If your last line could leave them thinking "so you never \
actually told me", you withheld rather than widened. Name it concretely - a \
decision not yet taken, a figure that does not add up, someone whose next move \
decides it - nameable in a handful of words, and worth a whole episode of its \
own.
- **It has to already be in the room.** Set it up in passing while you are \
telling the story and leave it standing. First mentioned in the final \
sentence, it reads as a tease.
- **Point at it, do not ask about it.** No rhetorical questions ("but will it \
hold?"), no promises about what comes next, no "we'll have to wait and see". \
State it as a fact still in motion and stop there.
- **No exit signals, ever:** "so, to sum up", "in conclusion", "all in all", \
"the bottom line is", "and that's the story of". Each hands the listener their \
coat.

The line you must not cross:
Every sentence carries information. Atmosphere on its own is cut. If a \
listener could ever think "get to the point", you have failed - the point \
arrives continuously, inside the story, from the first line to the last. Story \
is the shape of the delivery, never a delay before it.

Accuracy is part of being worth listening to:
- Never invent a statistic, quote, name, date or result. A story built on a \
made-up detail is worthless.
- If you do not know, say the short true thing and keep moving.
- If sources disagree, say so, and say which is better supported. Disagreement \
is usually the most interesting part anyway.
- Never fill a gap with something that merely sounds plausible. That is the \
worst thing you can do here.

Time, handled the way a person would:
- Give the newest information you can establish.
- Do NOT announce your own currency. No "as of Sunday the thirtieth", no \
"based on what I have".
- Mention timing only when it changes the meaning - "the count is still going" \
- and then in passing.
- Never narrate your own process, sourcing or uncertainty.

Format, because this is spoken aloud and never read:
- Output only the words to be said. No headings, markdown, bullets, stage \
directions, speaker labels or emoji.
- Flowing spoken English. Vary your sentence lengths - a short one lands a \
point. Say numbers as a person says them: "about twelve percent", "nineteen \
ninety-eight".
- No greeting, no sign-off, no naming the show, and never mention being an AI.

One line after the script, which is never spoken:

<<NEXT: the thing they would want to hear about next>>

Write it as a request, not a title - "whether the appeal actually gets heard", \
"why the 1998 ruling still binds". Six to twelve words, and it must be the \
thread you actually left open, not a related topic you thought of afterwards. \
The app strips it before anything is spoken; write nothing after it.
"""


def system_prompt() -> str:
    """The house rules, plus any examples of the voice they describe."""
    return SYSTEM_PROMPT + style_example_block()


@dataclass
class ScriptNotes:
    """What the model wrote that is not spoken.

    Passed in by the caller rather than kept on the generator. One generator
    serves many concurrent episodes, and per-episode state stashed on `self`
    has already caused one bug in this file; an argument cannot go stale.
    """

    #: The thread the episode deliberately left open, phrased as the follow-up
    #: a listener would ask for. Empty when the model did not name one.
    thread: str = ""


def extract_thread(text: str) -> str:
    """Pull the go-deeper thread out of the model's trailing marker line."""
    match = _NEXT_MARKER.search(text)
    if not match:
        return ""
    thread = re.sub(r"\s+", " ", match.group(1)).strip(" .\"'")
    return thread[:160]


@dataclass
class EpisodePlan:
    """The length contract for one episode."""

    query: str
    minutes: int
    target_seconds: int
    word_budget: int
    sections: list[str]
    #: Words already spoken by the cold open, deducted from the body's budget.
    reserved_words: int = 0
    #: What the listener has already heard, for a "go deeper" follow-up.
    context: str = ""
    #: Look up live sources. Costs 10-25s before the first word, so it is off
    #: unless the listener asked for something that genuinely needs today's facts.
    search: bool = False
    #: Replay only. A cache miss must fail rather than generate - Explore shows
    #: other people's finished episodes and must never spend a model call, so
    #: the guarantee lives in the pipeline rather than in the interface's good
    #: intentions.
    cached_only: bool = False
    #: Documents, photos and links the listener attached. An episode built on
    #: these is theirs alone: `pipeline` refuses to cache it, so it never
    #: reaches Explore or another listener.
    attachments: tuple = ()

    @property
    def images(self) -> list:
        return [a for a in self.attachments if getattr(a, "kind", "") == "image"]

    @property
    def readable(self) -> list:
        """Attachments that arrive as text in the prompt rather than as pixels."""
        return [a for a in self.attachments if getattr(a, "kind", "") != "image"]

    @property
    def body_budget(self) -> int:
        return max(30, self.word_budget - self.reserved_words)

    @property
    def min_words(self) -> int:
        return int(self.word_budget * 0.94)

    @property
    def max_words(self) -> int:
        return int(self.word_budget * 1.06)


def now_line() -> str:
    """The current date and time, so the script can anchor itself.

    Without this the model has no idea what "this morning" means and cannot
    tell a listener whether it is describing something current or stale.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).astimezone()
    return now.strftime("%A %d %B %Y at %H:%M %Z").replace(" 0", " ")


def plan_episode(
    query: str, minutes: int, context: str = "", search: bool | None = None,
    cached_only: bool = False, attachments: tuple = (),
) -> EpisodePlan:
    """Map a duration in minutes onto a concrete writing brief."""
    minutes = max(settings.min_minutes, min(settings.max_minutes, int(minutes)))
    target_seconds = minutes * 60
    word_budget = int(round(minutes * settings.target_wpm))

    # How much story this much time can hold. Not a template to fill - a note
    # on scope, so the model picks something it can actually resolve rather
    # than starting something too big and padding or truncating it.
    if minutes <= 2:
        sections = ["one question, opened and answered"]
    elif minutes <= 4:
        sections = ["one question, with the turn that makes it interesting"]
    elif minutes <= 7:
        sections = ["a question that turns two or three times before it resolves"]
    else:
        sections = ["the full arc, including how it came to be this way"]

    reserved = 18 if settings.enable_cold_open else 0
    use_search = settings.enable_web_search if search is None else bool(search)
    return EpisodePlan(
        query, minutes, target_seconds, word_budget, sections, reserved, context,
        use_search, cached_only, tuple(attachments or ()),
    )


def build_prompt(plan: EpisodePlan) -> str:
    budget = plan.body_budget
    # Empty unless a cold open is running. When one is, the listener has
    # already heard an opening line, so the script must not write its own -
    # without this the two stack up and the episode opens twice.
    already_opened = (
        "\nThe episode has ALREADY opened with one short framing sentence that "
        "the listener has heard. Do not write a greeting, a hook, or a restatement "
        "of the question - continue straight into substance.\n"
        if plan.reserved_words
        else ""
    )
    attached = ""
    if plan.readable or plan.images:
        blocks = "\n\n".join(a.as_prompt_block() for a in plan.readable)
        photos = len(plan.images)
        photo_line = ""
        if photos:
            photo_line = (
                f"\nThe listener also attached {photos} image"
                f"{'s' if photos != 1 else ''}, which you can see. Read "
                "what is actually in them.\n"
            )
        attached = f"""
The listener attached this themselves. It is the material they want the episode
built on, so it outranks anything you recall on the subject - where the two
disagree, theirs is the subject and you should say plainly that it differs from
what is generally reported. Do not pad with background they did not ask for,
and do not claim anything about a document beyond what is in it.

{blocks}
{photo_line}"""

    follow_up = ""
    if plan.context:
        follow_up = f"""
This is a FOLLOW-UP. The listener has just heard a briefing on:
<already_heard>{plan.context}</already_heard>

Treat that as known. Do not re-explain it or re-introduce the subject. Go
straight into the narrower thing they asked for and stay on it.
"""

    return f"""Someone just asked FAM this:

<request>{plan.query}</request>

It is currently {now_line()}. Prefer the newest information you can establish.
{attached}{follow_up}
You have about {plan.minutes} minute{"s" if plan.minutes != 1 else ""} - roughly
{budget} words. That is room for {plan.sections[0]}.

Answer them. They asked because they wanted to know something, and by the end
they must know it well enough to say it back in their own words. That is the
job; the rest is how it arrives.

{already_opened}
Pick a way in. Find the specific thing - the moment, the person, the number,
the detail - that makes this worth hearing, and start there. Then keep them
moving: each thing you tell them should make the next thing matter more. By the
end they should understand it, and should have felt taken somewhere rather than
briefed.

Then leave one thread open - and not the one they asked about. Close their
question first, completely. The thread is something the answer itself raised
and they could not have known to ask about when they started: a decision still
to be taken, a number that does not add up, someone whose next move decides it.
Set it up in passing while you are telling the story, then end pointed at it,
still open. Never a summary of what you just said, never a question asked of
the listener.

Then, on its own line after the script, write that thread as the follow-up they
would ask for:

<<NEXT: six to twelve words>>

That line is stripped before anything is spoken. Nothing goes after it.

The time is the listener's, not a quota. If the story resolves early, stop
there; a short piece that lands beats a long one padded out. If you catch
yourself saying a topic is complex, or restating something, the story is over -
end it.

Begin."""


def clean_for_speech(text: str) -> str:
    """Strip anything the model may have added that should not be spoken."""
    # The go-deeper marker, and any half-written one: everything from an
    # unmatched "<<" onwards is metadata, never speech.
    text = _NEXT_MARKER.sub("", text)
    text = re.sub(r"<<.*$", "", text, flags=re.S)
    # Stage directions first, while their brackets are still intact.
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)
    text = re.sub(r"\((?:music|sfx|pause|beat|sound)[^)]*\)", "", text, flags=re.I)
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"^\s*(?:host|narrator|intro|outro)\s*:\s*", "", text, flags=re.I | re.M)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


class ScriptGenerator:
    """Streams a length-controlled script out of Claude."""

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.anthropic_api_key
        # Built centrally so the HTTP version is pinned in one place; an empty
        # key still lets the SDK fall back to ANTHROPIC_AUTH_TOKEN or a stored
        # `ant auth login` profile.
        self.client = build_async_client(key)

    def _request_kwargs(self, plan: EpisodePlan) -> dict:
        # Photos travel as image blocks, not as text, and go *before* the
        # prompt: the instructions refer to them, so they have to be in view by
        # the time they are mentioned.
        content: list = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": image.media_type,
                           "data": image.data_b64},
            }
            for image in plan.images
        ]
        content.append({"type": "text", "text": build_prompt(plan)})
        kwargs: dict = {
            "model": settings.model,
            "max_tokens": settings.max_output_tokens,
            "system": system_prompt(),
            "output_config": {"effort": settings.effort},
            "messages": [{"role": "user", "content": content}],
        }
        if plan.search:
            kwargs["tools"] = [
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": settings.max_web_searches,
                }
            ]
        return kwargs

    async def stream_sentences(
        self, plan: EpisodePlan, notes: ScriptNotes | None = None
    ) -> AsyncIterator[str]:
        """Yield speech-ready sentences as Claude writes them.

        Sentence granularity is deliberate: it is the largest unit that keeps
        time-to-first-audio low, and the smallest unit that still gives the TTS
        engine enough context for natural intonation.
        """
        buffer = ""
        emitted_words = 0

        async with self.client.messages.stream(**self._request_kwargs(plan)) as stream:
            async for event in stream.text_stream:
                buffer += event
                # Everything from "<<" onwards is the go-deeper marker rather
                # than speech, and it can arrive split across events. Hold it
                # back instead of letting the sentence splitter reach it.
                speech, marker, rest = buffer.partition("<<")
                while True:
                    match = _SENTENCE_END.search(speech)
                    if not match:
                        break
                    sentence = clean_for_speech(speech[: match.end()])
                    speech = speech[match.end() :]
                    if sentence:
                        emitted_words += count_words(sentence)
                        yield sentence
                buffer = speech + marker + rest
                # Safety valve: a model that ignores the budget must not be
                # allowed to produce an hour of audio for a 1-minute request.
                if emitted_words > plan.max_words * 1.35:
                    break

            tail = clean_for_speech(buffer)
            if tail:
                yield tail
            if notes is not None:
                notes.thread = extract_thread(buffer)

            final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                detail = getattr(final, "stop_details", None)
                reason = getattr(detail, "explanation", None) or "the request was declined"
                yield clean_for_speech(f"I can't put together a briefing on that. {reason}")

    async def cold_open(
        self, plan: EpisodePlan, spoken_so_far: str = ""
    ) -> AsyncIterator[str]:
        """One framing sentence, written by a small fast model, no tools.

        This runs *concurrently* with the main researched call. Its only job is
        to be speakable within a few hundred milliseconds so the listener hears
        something while web search is still running.

        The prompt forbids any factual claim, because this model has done no
        research and must not guess ahead of what the main model will say. It
        frames the question; it never answers it.

        `spoken_so_far` is what the listener has already heard from earlier
        fills. Each fill used to be an independent call with an identical
        prompt, so a slow script produced several openings in a row that
        circled the same ground and did not follow from one another. Given what
        came before, a fill continues it instead of starting again.
        """
        avoid = ""
        if _RECENT_OPENERS:
            recent = "\n".join("- " + o for o in _RECENT_OPENERS[-5:])
            avoid = (
                "\nThese are the openings this listener has already heard today. "
                "Do not reuse their wording, their rhythm, or their opening move:\n"
                f"{recent}\n"
            )
        already = ""
        if spoken_so_far.strip():
            already = (
                "\nThe listener is already part-way into this episode. This is "
                "what they have heard so far, word for word:\n"
                f"<heard>{spoken_so_far.strip()[-1200:]}</heard>\n"
                "Continue directly from that last sentence. Do not open again, "
                "do not restate the subject, and do not circle back to a point "
                "it has already made - carry the same line of thought forward.\n"
            )
        prompt = (
            "A listener asked for a spoken briefing on this topic:\n"
            f"<topic>{plan.query}</topic>\n"
            f"{already}\n"
            f"Write up to four short sentences, {settings.cold_open_words} words in total, "
            "that open the episode. Each sentence must stand alone and make sense as "
            "the last thing said before the briefing proper begins, because playback "
            "may cut over to the main script after any one of them.\n\n"
            "Make it specific to THIS topic. Name the subject, and say what kind of "
            "question it is - what is unsettled about it, what a listener would want "
            "to know, why someone would be asking now. A listener who hears several "
            "of these should never feel they are hearing a template.\n\n"
            f"{avoid}\n"
            "Critical constraints:\n"
            "- State NO facts, figures, dates, names, results or opinions about the "
            "topic. You have done no research and anything you assert could be wrong. "
            "Frame the question; never answer it.\n"
            "- No greeting, no 'welcome to the show', no show name, no 'let's dive in'.\n"
            "- Do not say 'here is your briefing' or any variation. Start with substance "
            "about the shape of the question.\n"
            "- Output only the sentences. No <<NEXT>> line - you are opening "
            "the episode, not ending it."
        )
        try:
            message = await self.client.messages.create(
                model=settings.cold_open_model,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if message.stop_reason == "refusal":
                return
            text = clean_for_speech(" ".join(b.text for b in message.content if b.type == "text"))
            # Yield sentence by sentence: the pipeline stops taking them the
            # moment the real script is ready, so later ones are often unused.
            remember_opener(text)
            while text:
                match = _SENTENCE_END.search(text)
                if not match:
                    break
                piece, text = text[: match.end()].strip(), text[match.end() :]
                if piece:
                    yield piece
            if text.strip():
                yield text.strip()
        except Exception:
            # A missing cold open costs a second of latency, not the episode.
            log.warning("cold open failed; falling back to the main script", exc_info=True)

    async def top_up(
        self, plan: EpisodePlan, spoken_so_far: str, words_needed: int
    ) -> AsyncIterator[str]:
        """Ask for a short continuation when the episode is running short.

        Kept deliberately small and web-search-free: this call happens while the
        listener is already hearing audio, so latency matters more than depth,
        and the research has already been done by the first call.
        """
        words_needed = max(20, min(words_needed, 400))
        # Only the tail is sent back. The model needs to know where it is in the
        # episode, not to re-read the whole thing - and a short prompt is a
        # cheap prompt.
        tail = " ".join(spoken_so_far.split()[-120:])
        prompt = (
            f"You are continuing a spoken briefing about: {plan.query}\n\n"
            f"These were the last words spoken:\n<so_far>{tail}</so_far>\n\n"
            f"Write about {words_needed} more words that continue naturally from "
            "there, adding substance rather than restating what was said, and "
            "finish with a clean closing sentence. Spoken prose only - no "
            "headings, markdown or stage directions. Do not repeat the text above."
        )
        async with self.client.messages.stream(
            model=settings.model,
            max_tokens=min(settings.max_output_tokens, 2000),
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            buffer = ""
            async for event in stream.text_stream:
                buffer += event
                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break
                    sentence = clean_for_speech(buffer[: match.end()])
                    buffer = buffer[match.end() :]
                    if sentence:
                        yield sentence
            tail_sentence = clean_for_speech(buffer)
            if tail_sentence:
                yield tail_sentence

async def _demo() -> None:  # pragma: no cover - manual check
    plan = plan_episode("what is a heat pump", 2)
    gen = ScriptGenerator()
    async for sentence in gen.stream_sentences(plan):
        print(sentence)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo())
