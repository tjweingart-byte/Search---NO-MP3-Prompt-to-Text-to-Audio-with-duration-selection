"""The connection baseline: it must measure, and it must not spend anything."""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import socket

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "tools" / "tls_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("tls_baseline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_sends_no_request_and_uses_no_credential():
    """The whole claim of "free" rests on this.

    A handshake is a connection and nothing more: no HTTP verb, no API key, no
    Anthropic client. Checked on the source so it cannot drift into sending
    something billable.
    """
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    for billable in ("anthropic", "httpx", "httpx2", "requests", "urllib"):
        assert billable not in imported, f"the baseline imports {billable}"

    lowered = source.lower()
    for forbidden in ("api_key", "authorization", "x-api-key", "sendall", "\"post\"", "'post'"):
        assert forbidden not in lowered, f"the baseline references {forbidden!r}"


def test_a_handshake_reports_its_three_phases_separately():
    """DNS, TCP and TLS come apart here - the in-run tracer cannot split them."""
    module = _load()
    source = MODULE_PATH.read_text()
    body = source[source.index("def one_handshake"):source.index("def summarise")]
    # Each phase is bounded by its own perf_counter pair.
    assert body.count("time.perf_counter()") == 6
    for key in ("dns_seconds", "tcp_seconds", "tls_seconds", "total_seconds"):
        assert key in body
    assert "getaddrinfo" in body and "connect(" in body and "wrap_socket" in body


def test_a_refused_port_raises_rather_than_reporting_a_zero():
    module = _load()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(Exception):
        module.one_handshake("127.0.0.1", closed_port, timeout=1.0)


def test_the_socket_is_closed_even_when_tls_fails():
    """A failed handshake must not leak a socket across twenty trials."""
    module = _load()
    source = MODULE_PATH.read_text()
    body = source[source.index("def one_handshake"):source.index("def summarise")]
    assert "finally:" in body and "sock.close()" in body


def test_summarise_reports_the_spread_not_just_a_middle():
    module = _load()
    out = module.summarise([0.010, 0.030, 0.020])
    assert out == {"n": 3, "median": 0.020, "min": 0.010, "max": 0.030}
    assert module.summarise([]) is None


def test_the_cold_trial_is_kept_apart_from_the_warm_ones():
    """Trial 1 has a cold resolver cache. One median over both describes neither."""
    source = MODULE_PATH.read_text()
    assert "cold, warm = trials[0], trials[1:]" in source
    assert '"cold": cold' in source and '"warm"' in source
