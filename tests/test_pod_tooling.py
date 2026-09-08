"""The pod tooling, checked on a machine with no GPU and no credentials.

These exist because the tooling's whole job is to be trusted on hardware that
is being billed by the minute, where a wrong answer is expensive and a *plausible*
wrong answer is worse. Two things are worth proving here rather than there:

* the rights gate refuses, and refuses for the stated reason. A gate whose
  refusal path has never run is a decoration.
* the packer never copies identity onto rented hardware. The record the pod
  gets is regenerated from three fields, not copied, and this asserts that
  everything else is left behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.pack_for_pod as pack  # noqa: E402
import tools.pod_episode as episode  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

CLEARED = {
    "consent": "yes",
    "commercial_use": "yes",
    "synthetic_voice_cleared": "yes",
    "name": "A Real Person",
    "email": "someone@example.com",
    "recorded_by": "someone else",
    "notes": "recorded at the kitchen table, second take",
}


#: Deliberately NOT "reference_3.wav". These 104 bytes were once printed by a
#: test as `reference_3.wav  0 KB  sha256 7e79ea3ae003b903`, in the middle of a
#: RunPod gate's output, and were read as the real reference having been
#: overwritten. A fixture that shares the production voice's filename is
#: indistinguishable from it in any log that quotes the name.
FIXTURE_NAME = "fixture_voice"


@pytest.fixture
def reference(tmp_path):
    """A recording and a full rights record, as the packing machine has them."""
    wav = tmp_path / f"{FIXTURE_NAME}.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    (tmp_path / f"{FIXTURE_NAME}.rights.json").write_text(json.dumps(CLEARED))
    return wav


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A throwaway git repository standing in for the FAM checkout.

    The packer runs `git archive` and `git rev-parse` against `pack.ROOT`. This
    used to be the real repository, which made the test pass on a laptop and
    fail with exit 128 inside an extracted pod bundle - which carries no `.git`
    by design. Building a repository here tests the packer instead of the
    machine, and still exercises the real `git archive`.
    """
    root = tmp_path / "checkout"
    (root / "tools").mkdir(parents=True)
    (root / "app.py").write_text("# the app\n")
    (root / "tts.py").write_text("# the engines\n")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-must-not-ship\n")
    (root / "scripts.db").write_bytes(b"not source")
    (root / ".gitignore").write_text(".env\n*.db\n")
    for args in (["init", "-q", "-b", "main"],
                 ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "Test"],
                 ["add", "-A"], ["commit", "-q", "-m", "tree"]):
        subprocess.run(["git", "-C", str(root)] + args, check=True,
                       capture_output=True)
    monkeypatch.setattr(pack, "ROOT", root)
    return root


# --------------------------------------------------------------------------
# the rights gate
# --------------------------------------------------------------------------
def test_a_cleared_recording_passes(reference):
    checked = pack.check_reference(reference)
    assert checked["path"] == reference
    assert len(checked["sha256"]) == 64


@pytest.mark.parametrize("field", pack.RIGHTS_FIELDS)
def test_one_uncleared_field_stops_the_pack(reference, field):
    """Not two of three. Any one of them is a refusal."""
    record = dict(CLEARED, **{field: "no"})
    (reference.parent / f"{FIXTURE_NAME}.rights.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit) as exc:
        pack.check_reference(reference)
    assert field in str(exc.value), "the refusal must name what is not cleared"


def test_a_missing_record_stops_the_pack(reference):
    (reference.parent / f"{FIXTURE_NAME}.rights.json").unlink()
    with pytest.raises(SystemExit) as exc:
        pack.check_reference(reference)
    assert "rights record" in str(exc.value)


def test_a_missing_recording_stops_the_pack(tmp_path):
    with pytest.raises(SystemExit) as exc:
        pack.check_reference(tmp_path / "nothing.wav")
    assert "no reference recording" in str(exc.value)


def test_an_unreadable_record_is_refused_not_ignored(reference):
    """A record that will not parse must not read as a record that cleared."""
    (reference.parent / f"{FIXTURE_NAME}.rights.json").write_text("{not json")
    with pytest.raises(SystemExit) as exc:
        pack.check_reference(reference)
    assert "not valid JSON" in str(exc.value)


def test_the_record_is_found_where_older_tooling_wrote_it(tmp_path):
    """Beside the originals, one level up from a working folder."""
    working = tmp_path / "working"
    working.mkdir()
    wav = working / "reference_3.wav"
    wav.write_bytes(b"\0" * 32)
    (tmp_path / "reference_3.rights.json").write_text(json.dumps(CLEARED))
    assert pack.rights_for(wav).parent == tmp_path


# --------------------------------------------------------------------------
# identity does not leave the packing machine
# --------------------------------------------------------------------------
def test_the_pod_record_carries_the_answers_and_not_the_person(reference):
    minimal = pack.minimal_rights(CLEARED, reference)
    assert all(minimal[field] == "yes" for field in pack.RIGHTS_FIELDS)
    for field in pack.IDENTITY_FIELDS:
        assert field not in minimal, f"{field!r} would have reached the pod"


def test_the_record_is_rebuilt_not_copied(reference):
    """An unknown future field must not ride along by default."""
    minimal = pack.minimal_rights(dict(CLEARED, agreement_pdf="signed.pdf"),
                                  reference)
    assert "agreement_pdf" not in minimal
    assert set(minimal) == set(pack.RIGHTS_FIELDS) | {"reference", "note"}


def test_the_engine_would_accept_the_minimal_record(tmp_path, reference):
    """The point of the trim is that it still clears the production gate."""
    from tts import ChatterboxEngine

    pod = tmp_path / "pod-voice"
    pod.mkdir()
    (pod / "reference_3.wav").write_bytes(b"\0" * 32)
    (pod / "reference_3.rights.json").write_text(
        json.dumps(pack.minimal_rights(CLEARED, reference)))
    cleared, detail = ChatterboxEngine.rights_cleared(pod / "reference_3.wav")
    assert cleared, detail


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------
def test_the_bundle_ships_head_the_voice_and_nothing_secret(tmp_path, reference,
                                                            checkout, monkeypatch):
    out = tmp_path / "fam-pod.tar.gz"
    monkeypatch.setattr(pack, "OUT", out)
    assert pack.main(["--reference", str(reference)]) == 0

    with tarfile.open(out) as bundle:
        names = bundle.getnames()
        pod_txt = bundle.extractfile("FAM/POD.txt").read().decode()

    assert "FAM/app.py" in names and "FAM/tts.py" in names
    assert f"{pack.POD_VOICE_DIR}/{FIXTURE_NAME}.wav" in names
    assert f"{pack.POD_VOICE_DIR}/{FIXTURE_NAME}.rights.json" in names
    # git archive ships tracked files only, so none of this can be in there.
    # The throwaway checkout contains a real-looking .env and a .db precisely
    # so that "nothing secret" is a result rather than an absence of evidence.
    assert not [n for n in names if n.endswith((".env", ".git", "scripts.db"))]
    assert not [n for n in names if "/.git/" in n]
    assert "sk-ant-must-not-ship" not in pod_txt
    assert "ANTHROPIC_API_KEY" in pod_txt, "POD.txt must say where the key comes from"


def test_no_bundled_file_carries_the_checkouts_secret(tmp_path, reference,
                                                      checkout, monkeypatch):
    """Read every byte of the bundle, not just the file list."""
    out = tmp_path / "fam-pod.tar.gz"
    monkeypatch.setattr(pack, "OUT", out)
    assert pack.main(["--reference", str(reference)]) == 0
    with tarfile.open(out) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            body = bundle.extractfile(member).read()
            assert b"sk-ant-must-not-ship" not in body, member.name


def test_the_revision_and_voice_digest_reach_pod_txt(tmp_path, reference,
                                                     checkout, monkeypatch):
    """POD.txt is the pod's only record of what it is running: the bundle
    carries no .git, so this text is where the revision has to come from."""
    out = tmp_path / "fam-pod.tar.gz"
    monkeypatch.setattr(pack, "OUT", out)
    assert pack.main(["--reference", str(reference)]) == 0

    revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()[:16]
    with tarfile.open(out) as bundle:
        pod_txt = bundle.extractfile("FAM/POD.txt").read().decode()
    assert f"revision  {revision}" in pod_txt
    assert digest in pod_txt, "the pod cannot verify a voice it was not told about"
    assert pack.pod_txt_field(pod_txt, "revision") == revision


def test_packing_from_an_extracted_bundle_is_refused_with_the_reason(
        tmp_path, reference, monkeypatch):
    """Running the packer on the pod cannot work and must not look like a bug.

    The bundle carries no `.git` on purpose - a rented card is somewhere to run
    FAM for an hour, not somewhere to leave a token - so `git rev-parse` exited
    128 with nothing saying why. This is the sentence that replaces that.
    """
    extracted = tmp_path / "FAM"
    extracted.mkdir()
    (extracted / "POD.txt").write_text("revision  abc123\n")
    monkeypatch.setattr(pack, "ROOT", extracted)
    with pytest.raises(SystemExit) as exc:
        pack.main(["--reference", str(reference)])
    message = str(exc.value)
    assert "not a git checkout" in message
    assert "POD.txt" in message, "it must say where the revision actually is"


def test_uncommitted_work_stops_the_pack(tmp_path, reference, checkout,
                                         monkeypatch):
    """`git archive` ships HEAD, so a dirty tree would measure other code."""
    monkeypatch.setattr(pack, "OUT", tmp_path / "fam-pod.tar.gz")
    monkeypatch.setattr(pack, "working_tree_is_clean", lambda: (False, " M app.py"))
    with pytest.raises(SystemExit) as exc:
        pack.main(["--reference", str(reference)])
    assert "--allow-dirty" in str(exc.value)


# --------------------------------------------------------------------------
# reading the server's own timeline back
# --------------------------------------------------------------------------
def test_the_marks_are_read_from_the_line_the_server_writes(tmp_path):
    """`marks=` is the last field, so the JSON runs to the end of the line."""
    log = tmp_path / "server.log"
    log.write_text(
        'INFO episode q=\'a question\' {"sentences": 4} wall=9.1s marks='
        '{"claude_decoupled": true, "chunks": 5}\n')
    marks = episode.marks_from_log(log, 0, deadline=_soon())
    assert marks == {"claude_decoupled": True, "chunks": 5}


def test_a_previous_episodes_line_is_not_read_as_this_ones(tmp_path):
    log = tmp_path / "server.log"
    log.write_text('INFO episode q=\'old\' marks={"chunks": 1}\n')
    after = log.stat().st_size
    with log.open("a") as handle:
        handle.write('INFO episode q=\'new\' marks={"chunks": 2}\n')
    assert episode.marks_from_log(log, after, deadline=_soon())["chunks"] == 2


def test_no_line_at_all_is_none_rather_than_a_guess(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("INFO nothing relevant here\n")
    assert episode.marks_from_log(log, 0, deadline=_soon()) is None


def test_an_unread_timeline_is_reported_as_unverified(tmp_path, capsys):
    """Silence about the decoupling claim is the one thing not allowed."""
    run = {"headers": {}, "rate": 24000, "first_byte_seconds": 0.4,
           "total_seconds": 30.0, "audio_seconds": 180.0}
    assert episode.report(run, None) is False
    assert "unverified" in capsys.readouterr().out


def test_a_coupled_run_is_named_as_one(capsys):
    run = {"headers": {}, "rate": 24000, "first_byte_seconds": 0.4,
           "total_seconds": 30.0, "audio_seconds": 180.0}
    assert episode.report(run, {"claude_decoupled": False}) is False
    assert "COUPLED" in capsys.readouterr().out


def test_a_decoupled_run_is_the_only_success(capsys):
    run = {"headers": {}, "rate": 24000, "first_byte_seconds": 0.4,
           "total_seconds": 30.0, "audio_seconds": 180.0}
    assert episode.report(run, {"claude_decoupled": True}) is True
    assert "DECOUPLED" in capsys.readouterr().out


def _soon() -> float:
    import time

    return time.perf_counter() + 0.5


# --------------------------------------------------------------------------
# the shell script
# --------------------------------------------------------------------------
def test_the_pod_script_is_valid_shell():
    """A syntax error would surface on the pod, after the card started billing."""
    assert subprocess.run(["bash", "-n", str(ROOT / "tools" / "pod_production_test.sh")],
                          capture_output=True).returncode == 0


def test_the_pod_script_refuses_the_interim_voice():
    """The expensive mistake this gate exists to prevent: a full set of
    plausible numbers, measured on Piper."""
    source = (ROOT / "tools" / "pod_production_test.sh").read_text()
    assert 'report["interim"]' in source
    assert 'report["selected"] != "chatterbox"' in source


def test_the_pod_script_turns_the_cache_off_before_measuring():
    """A cached script reports a Claude time of nearly zero, and the comparison
    between legacy and phase6 would be between two replays."""
    source = (ROOT / "tools" / "pod_production_test.sh").read_text()
    measure = source.split("The measurement")[1]
    assert "CACHE_ENABLED=0" in measure
    assert measure.index("CACHE_ENABLED=0") < measure.index("run_pipeline legacy")


def test_the_pod_script_measures_both_pipelines_on_the_same_card():
    source = (ROOT / "tools" / "pod_production_test.sh").read_text()
    assert "run_pipeline legacy" in source and "run_pipeline phase6" in source


def test_the_chatterbox_requirements_keep_every_pin_that_cost_a_session():
    """Each of these was a silent failure that named the wrong cause. Removing
    one without reading its comment repeats the session it cost."""
    text = (ROOT / "requirements-chatterbox.txt").read_text()
    for pin in ("setuptools<82", "torchvision==0.21.0", "huggingface-hub>=1.3,<2",
                "tokenizers>=0.22,<=0.23"):
        assert pin in text, f"{pin} is what stops a silent failure; see its comment"
    assert "chatterbox-tts" in text
    # torch is deliberately unpinned here - chatterbox pins it, and the working
    # CUDA build is what that pin resolves to.
    assert "\ntorch\n" in text


def requirement_lines(path) -> list[str]:
    """The actual requirements, with comments and blanks dropped.

    Comments matter here: requirements.txt now *explains* which engine is
    installed separately and why, so a naive substring search over the whole
    file finds the words it is explaining.
    """
    return [line.split("#", 1)[0].strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_chatterbox_is_not_in_the_base_requirements():
    """Installing FAM, running its tests and building the preview must not
    require a multi-gigabyte deep-learning stack."""
    for requirement in requirement_lines(ROOT / "requirements.txt"):
        for package in ("chatterbox", "torch"):
            assert package not in requirement.lower(), requirement


def test_the_two_kinds_of_dirty_are_told_apart(reference, tmp_path, checkout,
                                               monkeypatch, capsys):
    """A modified tracked file and an untracked one are different hazards.

    The modified one is the dangerous case: it exists at HEAD in an older form,
    so the bundle carries a version of a file you are looking at a different
    version of, and nothing about the run looks wrong. An untracked file is
    usually another branch's working files - which can be irreplaceable, so the
    refusal says to write an ignore rule rather than to delete anything.
    """
    monkeypatch.setattr(pack, "OUT", tmp_path / "fam-pod.tar.gz")
    monkeypatch.setattr(pack, "working_tree_is_clean",
                        lambda: (False, " M app.py\n?? experiments/\n"))
    with pytest.raises(SystemExit):
        pack.main(["--reference", str(reference)])
    out = capsys.readouterr().out
    assert "modified, and tracked at HEAD:" in out and "app.py" in out
    assert "untracked:" in out and "experiments/" in out


def test_the_refusal_does_not_recommend_deleting_anything(reference, tmp_path,
                                                         checkout, monkeypatch):
    monkeypatch.setattr(pack, "OUT", tmp_path / "fam-pod.tar.gz")
    monkeypatch.setattr(pack, "working_tree_is_clean",
                        lambda: (False, "?? experiments/\n"))
    with pytest.raises(SystemExit) as exc:
        pack.main(["--reference", str(reference)])
    message = str(exc.value)
    assert "ignore rule" in message
    assert "irreplaceable" in message
    assert "not a way past this message" in message, (
        "--allow-dirty must not read as the fix")


def test_classify_splits_on_the_untracked_marker():
    changed, untracked = pack.classify(" M app.py\n?? .DS_Store\nA  new.py\n?? experiments/")
    assert changed == [" M app.py", "A  new.py"]
    assert untracked == ["?? .DS_Store", "?? experiments/"]


def test_the_branch_ignores_what_is_not_its_own(tmp_path):
    """`.DS_Store` and `experiments/` are what the Mac's tree was reporting.

    `experiments/` is tracked content of the experimental branch and is
    deliberately absent from production; what survives a branch switch is the
    part git never tracked even there, including the reference recordings. The
    ignore rule keeps those files exactly where they are - it is not a delete.
    """
    rules = (ROOT / ".gitignore").read_text()
    assert "\n.DS_Store\n" in rules
    assert "\nexperiments/\n" in rules
    assert "irreplaceable" in rules, (
        "the rule must say why the directory is not to be deleted")


def test_experiments_is_not_a_production_dependency():
    """The reason it can be absent at all: nothing production imports it."""
    for name in ("app.py", "pipeline.py", "tts.py", "script_generator.py",
                 "config.py", "speech_assembly.py", "script_buffer.py",
                 "episode_marks.py"):
        source = (ROOT / name).read_text()
        assert "import experiments" not in source
        assert "from experiments" not in source


# --------------------------------------------------------------------------
# the two shapes the server publishes the same timeline in
# --------------------------------------------------------------------------
def test_the_log_shape_is_read_as_the_timeline_it_is():
    """`to_dict()` nests the derived figures under "summary"; the header sends
    them flat. Reading one as the other found nothing and reported UNKNOWN -
    the one number the pod run exists to produce."""
    summary, chunks = episode.normalise({
        "events": {"claude_start": 0.0},
        "chunks": [{"index": 0, "generate_seconds": 0.3, "audio_seconds": 2.0}],
        "summary": {"claude_decoupled": True, "chunks": 1},
    })
    assert summary["claude_decoupled"] is True
    assert len(chunks) == 1


def test_the_header_shape_is_read_unchanged():
    summary, chunks = episode.normalise({"claude_decoupled": False, "chunks": 4})
    assert summary["claude_decoupled"] is False
    assert chunks == []


def test_no_marks_at_all_is_empty_rather_than_a_crash():
    assert episode.normalise(None) == ({}, [])


def test_a_run_read_from_the_log_still_reaches_a_verdict(capsys):
    """The regression itself: the log-shaped timeline must not come out
    UNKNOWN."""
    run = {"headers": {}, "rate": 24000, "first_byte_seconds": 0.4,
           "total_seconds": 30.0, "audio_seconds": 180.0}
    assert episode.report(run, {"events": {}, "chunks": [],
                                "summary": {"claude_decoupled": True}}) is True
    assert "DECOUPLED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# continuous playback
# --------------------------------------------------------------------------
def test_synthesis_staying_ahead_reads_as_continuous():
    """Each chunk makes 2s of audio in 0.3s, so the player never catches up."""
    chunks = [{"index": i, "generate_seconds": 0.3, "audio_seconds": 2.0}
              for i in range(5)]
    stall = episode.stall_analysis(chunks)
    assert stall["continuous"] is True
    assert stall["headroom_seconds"] == pytest.approx(1.7)
    assert stall["at_chunk"] == 1
    assert stall["final_lead_seconds"] == pytest.approx(8.8)


def test_a_chunk_slower_than_its_own_audio_starves_the_player():
    """The failure this exists to catch: generation falls behind playback and
    the listener hears silence mid-episode."""
    chunks = [{"index": 0, "generate_seconds": 0.3, "audio_seconds": 2.0},
              {"index": 1, "generate_seconds": 9.0, "audio_seconds": 2.0}]
    stall = episode.stall_analysis(chunks)
    assert stall["continuous"] is False
    assert stall["at_chunk"] == 1


def test_the_wait_before_the_first_word_is_not_counted_as_a_stall():
    """A stall is silence in the middle of an episode. The wait before the
    first word is time-to-first-listen, reported separately - counting it here
    would mark every episode starved and hide the failure this looks for."""
    stall = episode.stall_analysis(
        [{"index": 0, "generate_seconds": 5.0, "audio_seconds": 1.0}])
    assert stall["continuous"] is True
    assert stall["headroom_seconds"] is None, "one chunk cannot fall behind"


def test_a_slow_first_chunk_does_not_hide_a_later_stall():
    """The two must stay separable: a long first chunk is latency, and a slow
    second chunk is still a gap."""
    stall = episode.stall_analysis([
        {"index": 0, "generate_seconds": 5.0, "audio_seconds": 1.0},
        {"index": 1, "generate_seconds": 4.0, "audio_seconds": 2.0},
    ])
    assert stall["continuous"] is False
    assert stall["headroom_seconds"] == pytest.approx(-3.0)


def test_no_chunks_is_unknown_not_continuous():
    assert episode.stall_analysis([]) is None
