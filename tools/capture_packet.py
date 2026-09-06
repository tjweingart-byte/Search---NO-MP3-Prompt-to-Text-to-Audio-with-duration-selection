"""Capture one Exa evidence packet and save it, so benchmarks can replay it.

    python tools/capture_packet.py --name founder_ceos

One Exa call, roughly $0.005, no Claude call. It uses the recovered benchmark
parameters exactly - `type="fast"`, `num_results=8`, top 3 sources, 2
highlights each - so the saved packet is the same shape the verified
replication used.

Once saved, `experiments/adapters/packet.py` replays it with no network, which
is what lets a generation benchmark hold search constant instead of measuring
it. Re-running overwrites nothing: it refuses if the file exists, because a
packet that changed underneath a comparison would silently invalidate it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.adapters import packet as packet_mod   # noqa: E402
from experiments.registry import search_adapter          # noqa: E402
from experiments.timeline import Timeline                # noqa: E402

DEFAULT_QUERY = "Why are founder CEOs becoming harder for boards to remove?"


def capture_manifest(manifest_path: pathlib.Path, args) -> int:
    """Capture every packet a topics manifest lists, skipping ones already held.

    Skipping matters: a packet that changed underneath a comparison would
    invalidate it silently, and the founder_ceos packet in particular is the
    thread back to every earlier run.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    topics = manifest["topics"]
    exa = search_adapter("exa")

    pending = [t for t in topics if not packet_mod.packet_path(t["packet"]).exists()]
    held = len(topics) - len(pending)
    print(f"\n  manifest {manifest['name']}: {len(topics)} topics, "
          f"{held} already captured, {len(pending)} to fetch")
    if not pending:
        print("  Nothing to do.\n")
        return 0

    state = exa.available()
    if not state.ok:
        print(f"\n  Cannot capture: {state.reason}\n  {state.remedy}\n")
        return 2

    print(f"  {len(pending)} Exa call(s), about ${len(pending) * 0.005:.3f}\n")
    if not args.yes:
        reply = input("  Proceed? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("  Nothing captured.\n")
            return 1

    for topic in pending:
        result = asyncio.run(exa.search(topic["query"], Timeline()))
        record = {
            "query": topic["query"],
            "topic": topic["packet"],
            "category": topic["category"],
            "context": result.context,
            "sources": result.sources,
            "searches": result.searches,
            "captured_at": time.time(),
            "source": "exa",
            "manifest": manifest["name"],
        }
        path = packet_mod.packet_path(topic["packet"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"    {topic['packet']:<18} {len(result.context):>6} chars  "
              f"{len(result.sources)} sources  ({topic['category']})")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="founder_ceos")
    parser.add_argument("--manifest",
                        help="a topics manifest; captures every packet it lists")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--from-file", help="a text file holding a packet you already have")
    parser.add_argument("--force", action="store_true", help="overwrite an existing packet")
    parser.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args = parser.parse_args()

    if args.manifest:
        return capture_manifest(pathlib.Path(args.manifest), args)

    path = packet_mod.packet_path(args.name)
    if path.exists() and not args.force:
        print(f"\n  {path} already exists.")
        print("  Refusing to overwrite: a packet that changes underneath a")
        print("  comparison invalidates it silently. Use --force if you mean it.\n")
        return 1

    if args.from_file:
        text = pathlib.Path(args.from_file).read_text(encoding="utf-8")
        record = {"query": args.query, "context": text, "sources": [],
                  "captured_at": time.time(), "source": "pasted"}
    else:
        exa = search_adapter("exa")
        state = exa.available()
        if not state.ok:
            print(f"\n  Cannot capture: {state.reason}\n  {state.remedy}\n")
            return 2
        print(f"\n  One Exa call for: {args.query!r}")
        result = asyncio.run(exa.search(args.query, Timeline()))
        record = {
            "query": args.query,
            "context": result.context,
            "sources": result.sources,
            "searches": result.searches,
            "captured_at": time.time(),
            "source": "exa",
            "params": {k: v for k, v in result.detail.items() if k != "context"},
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  saved {path}")
    print(f"  {len(record['context'])} chars, {len(record['sources'])} sources\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
