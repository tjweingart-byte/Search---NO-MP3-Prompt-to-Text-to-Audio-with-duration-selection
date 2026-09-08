#!/usr/bin/env python3
"""One real episode against a running FAM server, measured three ways.

This is the validation gate itself: not a benchmark harness with a stub in it,
but the production server, the production engine, the production prompt, asked
a real question, with the audio written to a file you can listen to.

Three views of the same request, because each answers a question the others
cannot:

  listener    measured here, from the client. Time to the first byte and to
              the last. This is what someone with the app actually gets, and
              it is the only view that includes the network.
  preroll     the `X-*` response headers. The server's own view of everything
              up to the moment it decided there was enough audio to answer.
  episode     the server's complete timeline, read back from its log. Includes
              what the headers cannot: `claude_complete`, `speaking_complete`,
              the backlog when the script finished, and `claude_decoupled` -
              the Phase 6 claim, true only when synthesis began strictly
              before Claude finished writing.

The headers stop at the preroll because a StreamingResponse fixes its headers
before the body runs. That is a property of streaming, not an oversight, so the
log is read rather than the header being made to lie.

    python tools/pod_episode.py --log /workspace/server.log \\
        --out experiments/pod --minutes 3 "why do stock markets close"

Nothing here starts or stops a server, and nothing here spends on a GPU beyond
the one episode it was asked for.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

WAV_HEADER_BYTES = 44

#: How long to keep looking for the request's own log line after the stream
#: ends. The line is written in the generator's `finally`, so it can trail the
#: last byte by a moment.
LOG_GRACE_SECONDS = 10.0


def wav_header(rate: int, data_bytes: int) -> bytes:
    """A real header with real sizes - this file is finished, not a live stream."""
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", data_bytes))


def fetch(base: str, query: str, minutes: int, search: str,
          voice: str) -> dict:
    """Stream one episode, timing it from the client's side."""
    params = {"q": query, "minutes": str(minutes), "fmt": "pcm"}
    if search in ("0", "1"):
        params["search"] = search
    if voice:
        params["voice"] = voice
    url = f"{base.rstrip('/')}/api/audio?" + urllib.parse.urlencode(params)

    started = time.perf_counter()
    first_byte = None
    body = bytearray()
    try:
        with urllib.request.urlopen(url, timeout=600) as response:
            headers = dict(response.headers)
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                if first_byte is None:
                    first_byte = time.perf_counter() - started
                body += chunk
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"the server refused the request ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {url}: {exc.reason}")

    rate = int(headers.get("X-Sample-Rate") or 24000)
    return {
        "url": url,
        "headers": headers,
        "rate": rate,
        "pcm": bytes(body),
        "first_byte_seconds": first_byte,
        "total_seconds": time.perf_counter() - started,
        "audio_seconds": len(body) / (rate * 2) if rate else 0.0,
    }


def marks_from_log(log: pathlib.Path, after: int, deadline: float) -> dict | None:
    """The request's own timeline, from the line the server writes when done.

    `after` is where the log ended before the request started, so a previous
    episode's line cannot be mistaken for this one's. `marks=` is the last
    field of the format string, so everything past it is the JSON.
    """
    while time.perf_counter() < deadline:
        if log.exists():
            with log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(after)
                tail = handle.read()
            for line in reversed(tail.splitlines()):
                _, sep, payload = line.partition("marks=")
                if sep:
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        time.sleep(0.25)
    return None


def _seconds(value) -> str:
    return "-" if value is None else f"{float(value):.2f}s"


def report(run: dict, marks: dict | None) -> bool:
    headers = run["headers"]
    print("\nlistener   what someone with the app actually got")
    print(f"  first byte           {_seconds(run['first_byte_seconds'])}")
    print(f"  whole episode        {_seconds(run['total_seconds'])}")
    print(f"  audio delivered      {run['audio_seconds']:.1f}s at "
          f"{run['rate']} Hz")
    requested = headers.get("X-Requested-Seconds")
    if requested:
        over = run["audio_seconds"] - float(requested)
        print(f"  against the ceiling  asked {float(requested):.0f}s, "
              f"{'over' if over > 0 else 'under'} by {abs(over):.1f}s")

    print("\npreroll    the server's view up to its first response byte")
    print(f"  preroll target       {headers.get('X-Preroll-Seconds', '-')}s")
    print(f"  first PCM at         {_seconds(headers.get('X-First-PCM-Seconds') or None)}")
    print(f"  preroll satisfied    {_seconds(headers.get('X-Preroll-Satisfied-Seconds') or None)}")
    print(f"  chunks primed        {headers.get('X-Chunks-Primed', '-')}")
    print(f"  audio primed         {headers.get('X-Audio-Primed-Seconds', '-')}s")

    if marks is None:
        print("\nepisode    NOT READ. The server's log line did not arrive, so "
              "the decoupling\n           claim is unverified for this run. "
              "Pass --log pointing at the file\n           the server's output "
              "is redirected to.")
        return False

    print("\nepisode    the server's complete timeline")
    for label, key in (
        ("Claude first token", "claude_ttft"),
        ("first sentence", "claude_to_first_sentence"),
        ("sentence -> synthesis", "first_sentence_to_synthesis"),
        ("first synthesis took", "first_synthesis_seconds"),
        ("first PCM", "first_pcm"),
        ("Claude finished", "claude_total"),
        ("speaking finished", "speaking_total"),
    ):
        print(f"  {label:<22}{_seconds(marks.get(key))}")
    print(f"  first chunk           {marks.get('first_chunk_sentences', '-')} "
          f"sentence(s), {marks.get('first_chunk_words', '-')} words")
    print(f"  chunks                {marks.get('chunks', '-')}, "
          f"{marks.get('chunk_words', '-')} words spoken")
    backlog = marks.get("backlog_at_claude_complete")
    print(f"  backlog when Claude   {backlog if backlog is not None else '-'}"
          "  (chunks still queued)")

    decoupled = marks.get("claude_decoupled")
    print("\nverdict")
    if decoupled is True:
        print("  DECOUPLED    synthesis began before Claude finished writing.")
    elif decoupled is False:
        print("  COUPLED      synthesis did not begin until Claude had "
              "finished. Phase 6's whole\n               claim is that this "
              "does not happen; on this run it did.")
    else:
        print("  UNKNOWN      the timeline did not carry both marks.")
    return decoupled is True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="what to ask. A real question, not a fixture.")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--minutes", type=int, default=3)
    parser.add_argument("--search", default="", choices=["", "0", "1"],
                        help="force research on or off; omit to let the "
                             "question decide, as production does")
    parser.add_argument("--voice", default="")
    parser.add_argument("--log", default="",
                        help="the file the server's output is redirected to. "
                             "Without it the complete timeline cannot be read "
                             "and the decoupling claim stays unverified.")
    parser.add_argument("--out", default="",
                        help="write episode.wav and episode.json here, so the "
                             "run can be listened to and not just believed")
    args = parser.parse_args(argv)

    log = pathlib.Path(args.log).expanduser() if args.log else None
    after = log.stat().st_size if log and log.exists() else 0

    print(f"asking     {args.query!r} for {args.minutes} minute(s)")
    run = fetch(args.base, args.query, args.minutes, args.search, args.voice)
    marks = (marks_from_log(log, after,
                            time.perf_counter() + LOG_GRACE_SECONDS)
             if log else None)
    decoupled = report(run, marks)

    if args.out:
        out = pathlib.Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        wav = out / "episode.wav"
        wav.write_bytes(wav_header(run["rate"], len(run["pcm"])) + run["pcm"])
        (out / "episode.json").write_text(json.dumps({
            "query": args.query,
            "minutes": args.minutes,
            "listener": {
                "first_byte_seconds": run["first_byte_seconds"],
                "total_seconds": run["total_seconds"],
                "audio_seconds": run["audio_seconds"],
                "sample_rate": run["rate"],
            },
            "headers": {k: v for k, v in run["headers"].items()
                        if k.lower().startswith("x-")},
            "marks": marks,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote      {wav}  ({run['audio_seconds']:.0f}s of speech)")
        print(f"           {out / 'episode.json'}")
        print("\n  Listen to it. Nothing above says whether it sounds like FAM.")

    return 0 if decoupled else 1


if __name__ == "__main__":
    sys.exit(main())
