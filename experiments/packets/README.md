# Saved evidence packets

One real Exa packet, captured once and replayed on every trial of every arm.

That is what lets a generation benchmark isolate Claude: search stops taking
time and stops varying, so a difference between arms cannot be a difference in
the evidence they were given.

    python tools/capture_packet.py --name founder_ceos

One Exa call, about $0.005. Re-running refuses to overwrite - a packet that
changed underneath a comparison would invalidate it without anything failing.

Packets are git-ignored: they are captured data, not source, and a stale one
committed here would quietly become the thing everyone benchmarks against.
