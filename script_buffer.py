"""A cheap text buffer between reading Claude and speaking it.

`pipeline._start` currently pumps `stream_sentences` straight into an
`asyncio.Queue(maxsize=QUEUE_DEPTH)` that the synthesiser drains. When the
voice falls behind, that queue fills and `await queue.put(...)` stops the
*reader* - so synthesis throughput decides when the model appears to finish.
The Phase 6 RTX 4090 run measured the effect directly: with the queue in
front of the reader, a 12.5s Claude stream was reported as 66.7s, of which
64.3s was our own backpressure.

The fix is one buffer and a size unit:

    Claude stream
      -> reader        never touches the synthesis queue
      -> ScriptBuffer  bounded by CHARACTERS
      -> assembler     whole sentences -> speech-sized chunks
      -> caller        synthesises, and may be as slow as it likes

**Text is the cheap half.** A three-minute FAM episode is about 2,700
characters, so the default bound is roughly twenty of them - a real bound that
no realistic episode approaches, holding kilobytes where the audio for the same
script would hold hundreds of megabytes. Conflating the two is what coupled the
stages in the first place.

The caller consuming `assemble_chunks()` can be arbitrarily slow: it suspends
this generator, not the reader task behind it. If the reader is ever held up -
only possible by reaching the character bound - `ScriptBuffer.blocked_seconds`
records it, so the condition is measurable instead of invisible.

Nothing here imports `pipeline`, so `pipeline` may import this.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Optional

from speech_assembly import AssembledChunk, AssemblyPolicy, SpeechAssembler

#: The buffer's bound, in characters. About twenty three-minute episodes.
DEFAULT_MAX_CHARACTERS = 64_000

#: How often the assembler wakes with no new sentence, to run its timer and
#: headroom rules. Small enough to be invisible against `max_wait_seconds`.
ASSEMBLER_TICK = 0.05


class ScriptBuffer:
    """Sentences waiting to be spoken, bounded by characters rather than count.

    A sentence is never rejected for being long; the bound only decides when a
    *further* sentence has to wait. That keeps the reader's behaviour
    independent of how the model happens to punctuate.
    """

    def __init__(self, max_characters: int = DEFAULT_MAX_CHARACTERS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_characters = max_characters
        self._clock = clock
        self._queue: asyncio.Queue = asyncio.Queue()
        self._characters = 0
        self._room = asyncio.Event()
        self._room.set()
        #: Time the producer spent unable to advance because of this buffer.
        #: Expected to stay at zero; if it does not, the bound is too small.
        self.blocked_seconds = 0.0
        self.peak_characters = 0
        self.peak_sentences = 0

    # -- producer ---------------------------------------------------------
    async def put(self, sentence: str) -> None:
        """Add a sentence, waiting only if the character bound is reached."""
        while self._characters and self._characters + len(sentence) > self.max_characters:
            self._room.clear()
            started = self._clock()
            await self._room.wait()
            self.blocked_seconds += self._clock() - started
        self._characters += len(sentence)
        self._queue.put_nowait(sentence)
        self.peak_characters = max(self.peak_characters, self._characters)
        self.peak_sentences = max(self.peak_sentences, self._queue.qsize())

    def fail(self, exc: BaseException) -> None:
        """Hand a producer failure to the consumer, which raises it.

        Matching `pipeline._start`, which puts the exception on the queue
        rather than swallowing it.
        """
        self._queue.put_nowait(exc)

    def close(self) -> None:
        self._queue.put_nowait(None)

    # -- consumer ---------------------------------------------------------
    async def get(self) -> Optional[str]:
        """The next sentence, or None once the producer has closed."""
        item = await self._queue.get()
        if item is None:
            return None
        if isinstance(item, BaseException):
            raise item
        self._characters -= len(item)
        if self._characters + 1 <= self.max_characters:
            self._room.set()
        return item

    # -- state ------------------------------------------------------------
    @property
    def characters(self) -> int:
        return self._characters

    @property
    def sentences(self) -> int:
        return self._queue.qsize()


async def assemble_chunks(
    sentences: AsyncIterator[str],
    policy: Optional[AssemblyPolicy] = None,
    buffer: Optional[ScriptBuffer] = None,
    headroom: Optional[Callable[[], Optional[float]]] = None,
    clock: Callable[[], float] = time.monotonic,
    tick: float = ASSEMBLER_TICK,
) -> AsyncIterator[AssembledChunk]:
    """Read `sentences` into a buffer and yield speech-sized chunks from it.

    The reader runs as its own task, so a slow consumer suspends this generator
    and never the reader. That is the whole decoupling; everything else here is
    bookkeeping.

    `headroom` is an optional callable returning the listener's current playback
    headroom in seconds, or None when nothing is playing yet. It is what lets
    the assembler abandon batching when the listener is about to catch up; a
    caller with no playback model omits it and gets the word rules alone.
    """
    policy = policy or AssemblyPolicy()
    buffer = buffer if buffer is not None else ScriptBuffer(clock=clock)
    assembler = SpeechAssembler(policy=policy, clock=clock)

    def now() -> Optional[float]:
        return headroom() if headroom is not None else None

    async def read() -> None:
        try:
            async for sentence in sentences:
                if sentence and sentence.strip():
                    await buffer.put(sentence)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            buffer.fail(exc)
        finally:
            buffer.close()

    reader = asyncio.create_task(read())
    try:
        while True:
            try:
                sentence = await asyncio.wait_for(buffer.get(), tick)
            except asyncio.TimeoutError:
                for chunk in assembler.due(now()):
                    yield chunk
                continue
            if sentence is None:
                for chunk in assembler.flush():
                    yield chunk
                break
            for chunk in assembler.offer(sentence, now()):
                yield chunk
    finally:
        # A consumer that stops early - a cancelled request, a truncated
        # episode - must not leave the reader running against a dead stream.
        reader.cancel()
        try:
            await reader
        except (asyncio.CancelledError, Exception):
            pass
