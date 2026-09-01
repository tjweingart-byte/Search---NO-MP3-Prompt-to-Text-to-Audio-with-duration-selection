"""Does the listener fall silent after the opener's first sentence?

No API key: Claude is replaced by a generator whose researched call takes a
configurable time, and the opener answers in ~1s like Haiku does. The measure
is the only one that matters - audio produced against wall clock consumed.
"""
import asyncio, dataclasses, sys, time
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline as P
import script_generator as sg
from config import settings
from pipeline import PodcastPipeline
from script_generator import plan_episode
from tts import DebugEngine

YIELDED = []
SPOKEN = [0]
OPENER = ["There is a question here that has not settled yet.",
          "It is the kind of thing people argue about for a reason.",
          "What counts as an answer keeps moving.",
          "That is what makes it worth going through slowly."]

class FakeGen:
    def __init__(self, script_delay): self.script_delay = script_delay
    async def cold_open(self, plan):
        await asyncio.sleep(1.0)          # Haiku, no tools
        for s in OPENER:
            YIELDED.append(s)
            yield s
    async def stream_sentences(self, plan, notes=None):
        await asyncio.sleep(self.script_delay)   # Opus, possibly with search
        for i in range(60): yield f"Body sentence number {i} carrying real information."
    async def top_up(self, plan, spoken, words):
        return
        yield

_ORIG_SPEAK = PodcastPipeline._speak_one

async def _counting(self, sentence, pace, stats):
    if sentence in OPENER: SPOKEN[0] += 1
    async for c in _ORIG_SPEAK(self, sentence, pace, stats): yield c

PodcastPipeline._speak_one = _counting


async def probe(delay):

    patched = dataclasses.replace(settings, enable_cold_open=True)
    P.settings = patched; sg.settings = patched
    pipe = PodcastPipeline(generator=FakeGen(delay), engine=DebugEngine(), cache=None)
    plan = plan_episode("why is the sky blue", 3)
    YIELDED.clear(); SPOKEN[0]=0
    start = time.perf_counter(); produced = 0.0; worst = 0.0; at = 0.0
    async for chunk in pipe.stream_pcm(plan):
        produced += len(chunk) / 2 / 22050
        wall = time.perf_counter() - start
        if wall > produced:                      # the listener has run out of audio
            if wall - produced > worst: worst, at = wall - produced, produced
    print(f"  script took {delay:>4.1f}s -> worst silence {worst:5.2f}s at {at:5.1f}s in"
          f" | opener sentences written: {len(YIELDED)}, spoken: {SPOKEN[0]}")

async def main():
    for d in (2.0, 5.0, 10.0, 20.0, 30.0):
        await probe(d)

asyncio.run(main())
