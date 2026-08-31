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

import anthropic

from anthropic_client import build_async_client
from config import settings

log = logging.getLogger(__name__)

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
# Anything that would be read aloud as punctuation noise rather than speech.
_MARKDOWN = re.compile(r"[*_`#>\[\]]|^\s*[-•]\s+", re.MULTILINE)

SYSTEM_PROMPT = """You write short spoken briefings for someone who chose to \
listen. Their attention is the only thing you can waste, so waste none of it.

The first sentence is everything. Someone just asked a question and is waiting \
to hear the answer. Give it to them immediately - the actual answer, or the \
single most concrete fact about the topic. Never open by describing what you \
are about to say.

Banned openings, without exception: "Here's what I can tell you about...", \
"Let's talk about...", "This is a fascinating topic...", "There's a lot to \
unpack here...", "So,", and anything else that fills a second without saying \
something. If your first sentence would still make sense with a different topic \
substituted in, it is wrong. Delete it and start with the fact.

What makes one of these good:
- Prefer one exact detail to three general statements. Names, numbers, what \
someone actually said or did. Vagueness is the failure mode; if a sentence \
would survive being written about a different topic, cut it.
- Cut anything the listener could have guessed before pressing play. "This is a \
complex issue with many perspectives" tells them nothing.
- Follow the story where it actually goes. Do not impose a shape on it - some \
topics are one clear answer, some are an argument, some are a sequence of \
events. Let the material decide.
- Earn the ending. Land on something that changes how they think about it, or \
tells them what to watch for. Do not summarise what you just said.

Being accurate is part of being worth listening to:
- Never invent a statistic, quote, name, date or result. If you do not know, \
say so in the script - "the full results are not in yet" is a real sentence a \
listener can use.
- If sources disagree, say that, and say which is better supported.
- Do not fill a gap in your knowledge with something that sounds plausible. \
That is the single worst thing you can do here.

Time matters as much as fact:
- Give the newest information you can establish, and say what it is current as \
of - "as of this morning", "as of Friday night" - the first time it matters.
- If something changed during the day, say what it was and what it is now.
- If an event is unresolved or still running, say so plainly rather than \
implying a final result.

Format, because this is spoken aloud and never read:
- Output only the words to be said. No headings, markdown, bullets, stage \
directions, speaker labels or emoji.
- Flowing spoken English. Short sentences. Say numbers and symbols the way a \
person says them - "about twelve percent", "nineteen ninety-eight".
- No greeting, no sign-off, no "welcome back", no "let's dive in", no naming \
the show, and never mention being an AI or describe your own process.
"""


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
    query: str, minutes: int, context: str = "", search: bool | None = None
) -> EpisodePlan:
    """Map a duration in minutes onto a concrete writing brief."""
    minutes = max(settings.min_minutes, min(settings.max_minutes, int(minutes)))
    target_seconds = minutes * 60
    word_budget = int(round(minutes * settings.target_wpm))

    # Not a template to fill. A note on how much ground this much time can
    # honestly cover, so the model picks a scope rather than padding one out.
    if minutes <= 2:
        sections = ["one thing, answered properly"]
    elif minutes <= 4:
        sections = ["one thing, with the context that makes it make sense"]
    elif minutes <= 7:
        sections = ["a few connected threads, in whatever order the story wants"]
    else:
        sections = ["the full picture, including how it got this way"]

    reserved = 18 if settings.enable_cold_open else 0
    use_search = settings.enable_web_search if search is None else bool(search)
    return EpisodePlan(
        query, minutes, target_seconds, word_budget, sections, reserved, context, use_search
    )


def build_prompt(plan: EpisodePlan) -> str:
    budget = plan.body_budget
    per_section = max(35, budget // len(plan.sections))
    outline = "\n".join(
        f"{i + 1}. {name} (about {per_section} words)"
        for i, name in enumerate(plan.sections)
    )
    already_opened = (
        "\nThe episode has ALREADY opened with one short framing sentence that "
        "the listener has heard. Do not write a greeting, a hook, or a restatement "
        "of the question - continue straight into substance.\n"
        if plan.reserved_words
        else ""
    )
    follow_up = ""
    if plan.context:
        follow_up = f"""
This is a FOLLOW-UP. The listener has just heard a briefing on:
<already_heard>{plan.context}</already_heard>

Treat that as known. Do not re-explain it or re-introduce the subject. Go
straight into the narrower thing they asked for and stay on it.
"""

    return f"""Write a spoken briefing for this listener:

<request>{plan.query}</request>

It is currently {now_line()}. Prefer the newest information you can establish,
and say what your picture is current as of.
{follow_up}
How long: about {plan.minutes} minute{"s" if plan.minutes != 1 else ""},
which is roughly {budget} words. Use that as the shape of the thing - enough
time for {plan.sections[0]}.

That length is the listener's time, not a quota. Fill it with substance or
finish early; a shorter briefing that is entirely worth hearing beats a longer
one padded to length. If you find yourself explaining that a topic is
complicated, or restating something you already said, you have run out of
material - stop there instead.

Start with the most concrete thing you know about this. Begin now."""


def clean_for_speech(text: str) -> str:
    """Strip anything the model may have added that should not be spoken."""
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
        kwargs: dict = {
            "model": settings.model,
            "max_tokens": settings.max_output_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": settings.effort},
            "messages": [{"role": "user", "content": build_prompt(plan)}],
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

    async def stream_sentences(self, plan: EpisodePlan) -> AsyncIterator[str]:
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
                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break
                    sentence = clean_for_speech(buffer[: match.end()])
                    buffer = buffer[match.end() :]
                    if sentence:
                        emitted_words += count_words(sentence)
                        yield sentence
                # Safety valve: a model that ignores the budget must not be
                # allowed to produce an hour of audio for a 1-minute request.
                if emitted_words > plan.max_words * 1.35:
                    break

            tail = clean_for_speech(buffer)
            if tail:
                yield tail

            final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                detail = getattr(final, "stop_details", None)
                reason = getattr(detail, "explanation", None) or "the request was declined"
                yield clean_for_speech(f"I can't put together a briefing on that. {reason}")

    async def cold_open(self, plan: EpisodePlan) -> AsyncIterator[str]:
        """One framing sentence, written by a small fast model, no tools.

        This runs *concurrently* with the main researched call. Its only job is
        to be speakable within a few hundred milliseconds so the listener hears
        something while web search is still running.

        The prompt forbids any factual claim, because this model has done no
        research and must not guess ahead of what the main model will say. It
        frames the question; it never answers it.
        """
        avoid = ""
        if _RECENT_OPENERS:
            recent = "\n".join("- " + o for o in _RECENT_OPENERS[-5:])
            avoid = (
                "\nThese are the openings this listener has already heard today. "
                "Do not reuse their wording, their rhythm, or their opening move:\n"
                f"{recent}\n"
            )
        prompt = (
            "A listener asked for a spoken briefing on this topic:\n"
            f"<topic>{plan.query}</topic>\n\n"
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
            "- Output only the sentences."
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

    async def full_script(self, plan: EpisodePlan) -> str:
        """Non-streaming convenience path, used by /api/script and by tests."""
        parts = [s async for s in self.stream_sentences(plan)]
        return " ".join(parts)


async def _demo() -> None:  # pragma: no cover - manual check
    plan = plan_episode("what is a heat pump", 2)
    gen = ScriptGenerator()
    async for sentence in gen.stream_sentences(plan):
        print(sentence)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo())
