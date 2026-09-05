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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="founder_ceos")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--from-file", help="a text file holding a packet you already have")
    parser.add_argument("--force", action="store_true", help="overwrite an existing packet")
    args = parser.parse_args()

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
