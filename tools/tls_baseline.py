"""How long does it cost merely to reach Anthropic? No request, no tokens.

    python tools/tls_baseline.py                    # 10 handshakes
    python tools/tls_baseline.py --trials 20 --save

Opens a TCP connection to the API host, completes the TLS handshake, and
closes it. **No HTTP request is sent, no credential is used, and nothing is
billed** - this is the cost of the connection alone.

It exists because the in-run tracer cannot separate DNS from TCP: httpcore
resolves the host inside `connect_tcp` and emits no event between the two.
Raw sockets can, so this is the one place those two numbers come apart.

Read it with one caveat in mind: after the first handshake the OS resolver
cache is warm, so trial 1 is the honest cold-DNS figure and the rest are the
warm case. Both are reported, separately, rather than averaged into a number
that describes neither.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import ssl
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

BASELINES = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "baselines"
DEFAULT_HOST = "api.anthropic.com"


def one_handshake(host: str, port: int, timeout: float) -> dict:
    """DNS, TCP and TLS, timed apart. The connection is closed immediately."""
    out: dict = {"host": host, "port": port}

    started = time.perf_counter()
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    out["dns_seconds"] = time.perf_counter() - started
    family, socktype, proto, _, address = infos[0]
    out["peer"] = address[0]

    sock = socket.socket(family, socktype, proto)
    sock.settimeout(timeout)
    try:
        started = time.perf_counter()
        sock.connect(address)
        out["tcp_seconds"] = time.perf_counter() - started

        context = ssl.create_default_context()
        started = time.perf_counter()
        wrapped = context.wrap_socket(sock, server_hostname=host)
        out["tls_seconds"] = time.perf_counter() - started
        out["tls_version"] = wrapped.version()
        cipher = wrapped.cipher()
        out["cipher"] = cipher[0] if cipher else None
        wrapped.close()
    finally:
        try:
            sock.close()
        except OSError:
            pass

    out["total_seconds"] = out["dns_seconds"] + out["tcp_seconds"] + out["tls_seconds"]
    return out


def summarise(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=443)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--save", action="store_true",
                        help="write the result to experiments/baselines/")
    args = parser.parse_args()

    print(f"\n  {args.trials} TLS handshakes to {args.host}:{args.port}")
    print("  No HTTP request, no credential, nothing billed.\n")

    trials, failures = [], []
    for index in range(1, args.trials + 1):
        try:
            record = one_handshake(args.host, args.port, args.timeout)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            print(f"    {index:>3}  FAILED  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        record["index"] = index
        trials.append(record)
        print(f"    {index:>3}  dns {record['dns_seconds'] * 1000:7.1f} ms   "
              f"tcp {record['tcp_seconds'] * 1000:7.1f} ms   "
              f"tls {record['tls_seconds'] * 1000:7.1f} ms   "
              f"total {record['total_seconds'] * 1000:7.1f} ms")

    if not trials:
        print(f"\n  No handshake completed. {failures[0] if failures else ''}\n")
        return 1

    # Trial 1 carries a cold resolver cache; the rest do not. Reporting one
    # median over both would describe neither.
    cold, warm = trials[0], trials[1:]
    print(f"\n  cold (trial 1)   dns {cold['dns_seconds'] * 1000:.1f} ms  "
          f"tcp {cold['tcp_seconds'] * 1000:.1f} ms  "
          f"tls {cold['tls_seconds'] * 1000:.1f} ms  "
          f"total {cold['total_seconds'] * 1000:.1f} ms")
    if warm:
        print("  warm (rest)      " + "  ".join(
            f"{name} {summarise([t[f'{name}_seconds'] for t in warm])['median'] * 1000:.1f} ms"
            for name in ("dns", "tcp", "tls", "total")))
    print(f"\n  {cold.get('tls_version')} / {cold.get('cipher')}  peer {cold.get('peer')}")
    if failures:
        print(f"  {len(failures)} handshake(s) failed and are recorded as failures.")

    payload = {
        "host": args.host, "port": args.port, "captured_at": time.time(),
        "trials": trials, "failures": failures,
        "cold": cold,
        "warm": {name: summarise([t[f"{name}_seconds"] for t in warm])
                 for name in ("dns", "tcp", "tls", "total")} if warm else None,
        "note": ("No HTTP request was sent and no credential was used. Trial 1 is "
                 "the cold-resolver figure; later trials run against a warm OS "
                 "cache. This is the only place DNS and TCP come apart - the "
                 "in-run tracer measures them together."),
    }
    if args.save:
        BASELINES.mkdir(parents=True, exist_ok=True)
        path = BASELINES / f"tls_{args.host.replace('.', '_')}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  saved {path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
