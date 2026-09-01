"""A built-in sample script, used when the server has no Claude credentials.

The point of demo mode is to let someone verify the *audio* half of the system -
instant streaming playback and duration matching - without needing an API key.
Everything downstream of the model is real: the same sentence pipeline, the same
pacing controller, the same raw-PCM streaming. Only the writer is canned.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

SAMPLE_SENTENCES = [
    "This is a demonstration briefing, and every part of it is being spoken as it is produced.",
    "There is no audio file anywhere in this process.",
    "The words you are hearing were turned into sound a fraction of a second before they reached you.",
    "Here is what is actually happening behind this recording.",
    "A script is written one sentence at a time.",
    "As soon as the first sentence exists, it is handed to a speech engine.",
    "That engine produces raw sound samples, which are pushed straight down the connection to your browser.",
    "Your browser schedules each batch of samples on an audio clock and plays them immediately.",
    "Nothing is ever encoded, and nothing is ever saved to disk.",
    "That is why playback begins almost at once, instead of after the whole episode has been built.",
    "The length of this briefing is not an accident either.",
    "Before every single sentence, the system checks how much time has already been used.",
    "It compares that against how many words are still waiting to be spoken.",
    "Then it adjusts the speaking rate so that the finish line arrives exactly when it should.",
    "If the script runs long, it is trimmed cleanly at the end of a sentence.",
    "If it runs short, more material is requested to fill the gap.",
    "The result lands within about one second of the length you selected.",
    "You can test that right now by choosing a different number of minutes and listening again.",
    "A one minute briefing and a ten minute briefing both hit their mark.",
    "When this runs with real credentials, the words come from Claude instead of this fixed sample.",
    "Claude researches the question using live web search, then writes to a strict word budget.",
    "Everything you are hearing now, the timing and the instant playback, works exactly the same way.",
    "The only difference is who wrote the sentences.",
    "If two different people ask for the same briefing, the second one reuses the first one's script.",
    "That costs nothing and starts playing almost instantly.",
    "This sample will repeat itself if you ask for a longer briefing than it can fill.",
    "That repetition is a limitation of the demo text, not of the system.",
    "So if the timing sounds right and the audio started quickly, the approach is working.",
]


class DemoGenerator:
    """Stands in for ScriptGenerator when there are no credentials."""

    #: Mirrors the real generator's interface so the pipeline cannot tell them apart.
    client = None

    async def stream_sentences(self, plan, notes=None) -> AsyncIterator[str]:
        # Demo mode leaves a thread too, so the Go Deeper suggestion can be
        # exercised without credentials.
        if notes is not None:
            notes.thread = f"how {plan.query} actually works underneath"
        # A small delay imitates the model streaming, so demo mode exercises the
        # same producer/consumer path as a real generation rather than dumping
        # every sentence in at once.
        budget = plan.body_budget
        spoken = 0
        index = 0
        while spoken < budget:
            sentence = SAMPLE_SENTENCES[index % len(SAMPLE_SENTENCES)]
            index += 1
            spoken += len(sentence.split())
            await asyncio.sleep(0.02)
            yield sentence

    async def top_up(self, plan, spoken_so_far: str, words_needed: int) -> AsyncIterator[str]:
        spoken = 0
        index = 0
        while spoken < words_needed:
            sentence = SAMPLE_SENTENCES[index % len(SAMPLE_SENTENCES)]
            index += 1
            spoken += len(sentence.split())
            yield sentence
