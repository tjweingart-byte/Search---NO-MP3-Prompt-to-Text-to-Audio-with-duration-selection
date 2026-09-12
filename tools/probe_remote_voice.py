"""What the remote voice worker actually exposes, and whether it speaks.

Written because a production 404 could not be told apart from three different
causes without asking the worker directly:

    POST https://<pod>-8002.proxy.runpod.net/synth -> 404 Not Found

is what you see whether the worker serves a different route, whether the
proxied port has nothing behind it, or whether the pod is running the
serverless handler and opening no port at all. A 404 says only "nobody was
home at that address", and the address has three halves that can be wrong.

So this asks, in order, and prints what came back:

    GET  /health         what the worker says it can do
    GET  /openapi.json   every route it serves, which is the authority on the
                         route name - not this repo, which may be a different
                         version than the image
    POST <synth route>   a real sentence, decoded through the same checks the
                         app applies, because "a route exists" is not "it
                         speaks" (PROBLEMS.md §52)

Exit code 0 only when real audio came back. 1 means something answered but
cannot speak; 2 means nothing at that address is a FAM voice worker.

    python tools/probe_remote_voice.py
    python tools/probe_remote_voice.py --url https://<pod>-8001.proxy.runpod.net
    REMOTE_VOICE_TOKEN=... python tools/probe_remote_voice.py --text "One line."

The token is read from the environment or the credential chain like everywhere
else, and is never printed.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remote_voice
from config import settings

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m")


def _say(mark: str, colour: str, line: str) -> None:
    print(f"  {colour}{mark}{RESET} {line}")


def probe(base: str, token: str, text: str, timeout: float) -> int:
    import httpx

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"\n{BOLD}Remote voice worker{RESET}  {base}")
    print(f"{DIM}  token: {'sent' if token else 'none - an open port'}{RESET}\n")

    answered = False
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        # 1. /health. Cheap, and the only call that is safe to poll.
        try:
            health = client.get(f"{base}/health", headers=headers)
        except Exception as exc:
            _say("x", RED, f"GET /health did not connect: {type(exc).__name__}: {exc}")
            return 2
        if health.status_code == 200:
            answered = True
            _say("ok", GREEN, f"GET /health 200 {health.text[:200]}")
        else:
            _say("x", RED, f"GET /health {health.status_code} {health.text[:200]}")

        # 2. The route table, which is the question this tool exists for.
        routes: dict = {}
        try:
            schema = client.get(f"{base}/openapi.json", headers=headers)
            if schema.status_code == 200:
                answered = True
                routes = (schema.json() or {}).get("paths") or {}
            else:
                _say("x", RED, f"GET /openapi.json {schema.status_code}")
        except Exception as exc:
            _say("x", RED, f"GET /openapi.json failed: {type(exc).__name__}: {exc}")
        if routes:
            for path, methods in sorted(routes.items()):
                verbs = ",".join(sorted(m.upper() for m in methods))
                _say("-", DIM, f"{verbs} {path}")
        if not answered:
            print(f"\n{RED}Nothing at {base} is a FAM voice worker.{RESET}")
            print("  Two things cause this, and both are on the pod:")
            print("   * the proxied port has nothing behind it. Dockerfile.voice")
            print("     serves ${PORT:-8001}, so a URL naming another port needs")
            print("     PORT set to it in the pod's environment.")
            print("   * VOICE_WORKER_MODE is not http, so the container is running")
            print("     the serverless handler, which opens no port at all.")
            return 2

        # 3. Make it speak. The only question that cannot be answered by reading.
        route = remote_voice.SYNTH_ROUTE
        posts = [p for p, m in routes.items() if "post" in m]
        if route not in routes and len(posts) == 1:
            route = posts[0]
            _say("!", YELLOW, f"no {remote_voice.SYNTH_ROUTE}; using POST {route}")
        payload = {"text": text, "voice": settings.remote_voice_id or None,
                   "sample_rate": settings.remote_voice_sample_rate,
                   "format": remote_voice.WIRE_FORMAT}
        try:
            reply = client.post(f"{base}{route}", json=payload, headers=headers)
        except Exception as exc:
            _say("x", RED, f"POST {route} failed: {type(exc).__name__}: {exc}")
            return 1
        if reply.status_code != 200:
            _say("x", RED, f"POST {route} {reply.status_code} {reply.text[:300]}")
            return 1
        try:
            config = remote_voice.RemoteChatterboxEngine.config()
            pcm = remote_voice.RemoteChatterboxEngine._pcm_from(reply.json(), config)
        except Exception as exc:
            _say("x", RED, f"POST {route} answered, but not with playable audio: {exc}")
            return 1
        seconds = len(pcm) / float(config.sample_rate * settings.sample_width)
        _say("ok", GREEN, f"POST {route} 200: {len(pcm)} bytes, "
                          f"{seconds:.1f}s at {config.sample_rate} Hz")
    if route != remote_voice.SYNTH_ROUTE:
        print(f"\n{YELLOW}This worker speaks, but at POST {route}.{RESET} The app "
              f"asks for {remote_voice.SYNTH_ROUTE} first and falls back to what "
              "the worker names; rebuild the image from Dockerfile.voice to stop "
              "paying an extra request per episode.\n")
        return 0
    print(f"\n{GREEN}This worker speaks.{RESET} "
          f"REMOTE_VOICE_URL={base} is correct.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="",
                        help="the pod's base URL; defaults to REMOTE_VOICE_URL")
    parser.add_argument("--text", default="Testing the remote voice.",
                        help="one short sentence to synthesise")
    parser.add_argument("--timeout", type=float,
                        default=settings.remote_voice_timeout,
                        help="seconds to allow the synthesis")
    args = parser.parse_args()

    config = remote_voice.RemoteChatterboxEngine.config()
    # An endpoint id is not an address: the serverless transport is a queue API
    # with no /health and no route table, and probing it as if it were a pod
    # would report the wrong thing about the wrong machine.
    inherited = config.base_url() if config.transport == "http" else ""
    base = (args.url or inherited).rstrip("/")
    if base.endswith(remote_voice.SYNTH_ROUTE):
        base = base[:-len(remote_voice.SYNTH_ROUTE)]
    if not base:
        print("This probe needs the address of an always-on worker: pass --url, "
              "or set REMOTE_VOICE_TRANSPORT=http and REMOTE_VOICE_URL. A "
              "serverless endpoint is checked with verify_voice.py instead - it "
              "is a queue, not a port. See REMOTE_VOICE.md.", file=sys.stderr)
        return 2
    token = config.api_key or os.environ.get("REMOTE_VOICE_TOKEN", "")
    return probe(base, token.strip(), args.text, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
