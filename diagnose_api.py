"""Work out why the Anthropic API is unreachable.

`APIConnectionError: Connection error` hides the cause. This reports the HTTP
configuration actually in force, the proxy environment, and what happens on a
real request - so the failure can be attributed rather than guessed at.

    python diagnose_api.py
"""
from __future__ import annotations

import asyncio
import os
import ssl
import sys

import anthropic

from anthropic_client import build_async_client, describe_http_version, http2_enabled
from config import settings


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


async def main() -> int:
    section("Versions")
    print(f"  anthropic        : {anthropic.__version__}")
    try:
        import httpx2

        print(f"  httpx2           : {httpx2.__version__}  (the SDK's HTTP library)")
    except ImportError:
        print("  httpx2           : MISSING - the SDK cannot work")
    try:
        import h2

        print(f"  h2               : {h2.__version__}  (HTTP/2 is possible)")
    except ImportError:
        print("  h2               : not installed - HTTP/2 cannot be negotiated")
    print(f"  python           : {sys.version.split()[0]}")
    print(f"  openssl          : {ssl.OPENSSL_VERSION}")

    section("HTTP configuration")
    print(f"  pinned to        : {describe_http_version()}")
    print(f"  http2 on pool    : {http2_enabled()}")
    print(f"  ANTHROPIC_HTTP2  : {os.environ.get('ANTHROPIC_HTTP2', '(unset -> HTTP/1.1)')}")

    section("Environment")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        value = os.environ.get(name)
        if name.endswith(("KEY", "TOKEN")) and value:
            value = f"set ({len(value)} chars, ends {value[-4:]})"
        print(f"  {name:18}: {value or '(unset)'}")
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "NO_PROXY", "no_proxy"):
        if os.environ.get(name):
            print(f"  {name:18}: {os.environ[name]}")
    print(f"  SSL_CERT_FILE     : {os.environ.get('SSL_CERT_FILE', '(unset)')}")
    print(f"  REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', '(unset)')}")

    section("Live request")
    if not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("  No credentials found; skipping. Set ANTHROPIC_API_KEY and re-run.")
        return 1

    client = build_async_client()
    try:
        message = await client.messages.create(
            model=settings.model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        text = " ".join(b.text for b in message.content if b.type == "text")
        print(f"  SUCCESS over {describe_http_version()} - model replied: {text.strip()!r}")
        return 0
    except anthropic.APIConnectionError as exc:
        print(f"  FAILED to connect: {exc}")
        cause = exc.__cause__
        while cause is not None:
            print(f"    caused by: {type(cause).__name__}: {cause}")
            cause = getattr(cause, "__cause__", None)
        print(
            "\n  The chain above names the real cause. Common ones:\n"
            "    ConnectError / getaddrinfo   -> DNS or no outbound route\n"
            "    SSLCertVerificationError     -> TLS interception; set SSL_CERT_FILE\n"
            "    ProxyError / 403 from proxy  -> the proxy is refusing CONNECT\n"
            "    ConnectTimeout               -> traffic is being dropped, not refused"
        )
        return 2
    except anthropic.AuthenticationError as exc:
        print(f"  Reached the API, but the key was rejected: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"  Reached the API but the call failed: {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
