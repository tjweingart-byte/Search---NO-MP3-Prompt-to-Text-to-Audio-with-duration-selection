"""Can a person actually paste the key? Answered with a real terminal.

The bug that made this necessary: `read -r answer` returns immediately at end
of input, so with stdin not a terminal the question printed and was declined in
the same breath - indistinguishable, from the outside, from never being asked.

It had a test. The test piped "n" in and checked the declined branch worked,
which proves that branch works and says nothing about whether anyone can reach
the other one. **Testing a prompt without a terminal tests everything except
the prompt.** The fix for that is not a better assertion, it is a pty.

The second fault these cover: a key WellSaid accepted was being thrown away
because this machine could not decode the reply. The key is good either way,
and having to paste it again after installing ffmpeg is the failure.
"""
from __future__ import annotations

import os
import pathlib
import re
import select
import sys
import tempfile
import textwrap
import time

import pytest

pytest.importorskip("pty", reason="no pseudo-terminals on this platform")
import pty  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: What a fake WellSaid returns, and whether this machine can decode it.
SCENARIOS = {
    # Their documented format is MP3; WAV needs no decoder at all.
    "wav": ("audio/wav", True),
    # The one that discarded a good key: WellSaid answered, ffmpeg was missing.
    "mp3": ("audio/mpeg", False),
}


def _fake_httpx(kind: str) -> pathlib.Path:
    """A sitecustomize that installs a fake httpx into the child process.

    The child is a real subprocess with a real terminal, so it cannot be
    monkeypatched from here - the stub has to be on its import path.
    """
    shim = pathlib.Path(tempfile.mkdtemp()) / "sitecustomize.py"
    shim.write_text(textwrap.dedent(f'''
        import array, struct, sys, types
        def wav(pcm, rate=22050):
            return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt "
                    + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                    + b"data" + struct.pack("<I", len(pcm)) + pcm)
        BODY = (wav(array.array("h", [6000] * 22050).tobytes())
                if {kind!r} == "audio/wav"
                else b"ID3\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\xff\\xfb" + b"\\x00" * 400)
        class R:
            status_code = 200
            headers = {{"Content-Type": {kind!r}}}
            text = ""
            content = BODY
        class C:
            def __init__(self, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, *a, **k): return R()
        m = types.ModuleType("httpx"); m.AsyncClient = C
        sys.modules["httpx"] = m
    '''))
    return shim


def drive(kind: str, decodable: bool, env_file: str, key: str) -> tuple[bool, str]:
    """Run setup_wellsaid.py in a real terminal; type `key` when asked."""
    shim = _fake_httpx(kind)
    env = dict(
        os.environ,
        FAM_ENV_FILE=env_file,
        PYTHONPATH=str(shim.parent),
        # Hiding ffmpeg is how "this machine cannot decode it" is produced.
        PATH=os.environ.get("PATH", "") if decodable else "/nonexistent:/usr/bin:/bin",
    )
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child execs immediately
        os.chdir(ROOT)
        os.execve(sys.executable, [sys.executable, "setup_wellsaid.py"], env)

    seen, asked, deadline = "", False, time.time() + 60
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if not ready:
            continue
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        seen += data.decode("utf-8", "replace")
        if not asked and "WellSaid API key" in seen:
            asked = True
            os.write(fd, key.encode() + b"\n")
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    return asked, re.sub(r"\033\[[0-9;]*m", "", seen)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_prompt_stops_and_waits_for_a_real_person(scenario, tmp_path):
    """The whole bug in one assertion: does it actually stop and ask?"""
    kind, decodable = SCENARIOS[scenario]
    env_file = str(tmp_path / "env")
    asked, output = drive(kind, decodable, env_file, "ws-a-genuinely-good-key")
    assert asked, f"the prompt was never reached:\n{output[-800:]}"


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_a_key_wellsaid_accepted_is_stored_even_if_this_machine_cannot_play_it(
        scenario, tmp_path):
    """Two questions, two answers. "Is the key good" is WellSaid's to answer;
    "can this machine decode the reply" is a local problem with a local fix,
    and it must not cost you the key."""
    kind, decodable = SCENARIOS[scenario]
    env_file = str(tmp_path / "env")
    _, output = drive(kind, decodable, env_file, "ws-a-genuinely-good-key")

    stored = pathlib.Path(env_file).read_text() if os.path.exists(env_file) else ""
    assert "WELLSAID_API_KEY=ws-a-genuinely-good-key" in stored, (
        f"a key WellSaid accepted was not stored ({scenario}):\n{output[-800:]}")
    # Stored with 600 permissions, like the Anthropic one.
    assert oct(pathlib.Path(env_file).stat().st_mode)[-3:] == "600"

    if not decodable:
        # Stored, and still honest that it cannot be heard yet.
        assert "ONE MORE STEP" in output
        assert "ffmpeg" in output
        assert "not have to paste it again" in output


def test_the_key_is_never_echoed_into_the_scrollback(tmp_path):
    """getpass, so a screenshot or a shared terminal log cannot leak it."""
    env_file = str(tmp_path / "env")
    _, output = drive("audio/wav", True, env_file, "ws-secret-value-here")
    assert "ws-secret-value-here" not in output, "the key was echoed to the terminal"
