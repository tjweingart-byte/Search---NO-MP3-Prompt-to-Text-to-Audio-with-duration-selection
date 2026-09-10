"""Everything the shipped code imports is either declared or deliberately not.

`numpy` was imported by `tts.pcm_from_float` on the synthesis path and declared
nowhere. It passed on every machine anyone developed on, because torch and
piper had both pulled numpy in as a transitive dependency at some point, and it
failed the moment CI installed `requirements.txt` and nothing else: six tests,
one `ModuleNotFoundError`, on the one environment that is a clean install.

That is the same shape as PROBLEMS.md 54 (`.env.example` disagreeing with
`config.py`) and 64 (a path resolved against the working directory): the thing
was true where it was written and false where it ran, and nothing compared the
two. `tests/test_env_example.py` closed the first. This closes this one.

The rule is not "every import must be declared" - several genuinely should not
be. Chatterbox and torch are gigabytes and belong in
`requirements-chatterbox.txt`; a sentence-embedding model is optional by
design. The rule is that an undeclared import must be a **decision**, written
down here with its reason, so that adding an eighth one is a deliberate
two-line change rather than an invisible one.
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Imported without being declared, on purpose. Each entry needs a reason,
#: because the whole value of this list is that adding to it is uncomfortable.
DELIBERATELY_OPTIONAL = {
    # The production voice and its stack: gigabytes, GPU-only, and installed
    # from requirements-chatterbox.txt. Imported inside the engine, which
    # raises TTSUnavailable when they are absent, so the app starts and says
    # what it cannot do rather than failing to import.
    "chatterbox": "requirements-chatterbox.txt",
    "torch": "requirements-chatterbox.txt",
    # A local sentence embedder, if anyone installs one. embeddings.py logs
    # loudly and falls back to the lexical backend when they are missing, so
    # near matching degrades rather than breaking. PROBLEMS.md 68.
    "onnxruntime": "optional; embeddings.py falls back and says so",
    "tokenizers": "optional; embeddings.py falls back and says so",
    # The Exa research backend, installed from requirements-exa.txt and needed
    # only when RESEARCH_BACKEND=exa. The default backend is `claude`, which
    # uses Anthropic's server-side search and needs nothing here. Imported
    # inside research._client, which raises ResearchUnavailable naming the
    # missing package, and research.diagnose() reports it on /api/health - so
    # the app starts and says what it cannot do.
    "exa_py": "requirements-exa.txt",
    # Sign in with Google and Sign in with Apple, installed from
    # requirements-oauth.txt. Imported inside `oauth._library`, which raises
    # OAuthUnavailable naming the pip line; /api/health reports it per
    # provider, and a sign-in attempt is refused with that reason.
    #
    # The load-bearing part is what it does *not* do: with this missing there
    # is no path that accepts a token it could not verify. An unverified JWT is
    # not a weak credential, it is anybody's credential, so this is the one
    # optional dependency whose absence must never degrade into a fallback.
    "jwt": "requirements-oauth.txt",
    # diagnose_api.py reports which HTTP libraries are present. Both imports
    # are inside try/except and their absence IS the diagnostic output.
    "h2": "diagnose_api.py reports its absence rather than needing it",
    "httpx2": "diagnose_api.py reports its absence rather than needing it",
}


def declared() -> set:
    """Package names in requirements.txt, normalised."""
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        for sep in ("[", ">", "<", "=", "!", "~", ";"):
            line = line.split(sep)[0]
        if line.strip():
            names.add(line.strip().lower().replace("_", "-"))
    return names


def imported() -> dict:
    """Third-party top-level imports in the shipped modules -> where from.

    Root modules only. `tools/` and `tests/` are developer surface; the
    question here is what a deployed copy of the app needs in order to run.
    """
    local = {p.stem for p in ROOT.glob("*.py")}
    standard = set(sys.stdlib_module_names)
    out: dict[str, set] = {}
    for path in sorted(ROOT.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # Relative imports name nothing to install.
                names = ([node.module.split(".")[0]]
                         if node.module and node.level == 0 else [])
            else:
                continue
            for name in names:
                if name in standard or name in local or name == "__future__":
                    continue
                out.setdefault(name, set()).add(path.name)
    return out


def normalise(name: str) -> str:
    """One spelling for a package, whichever side of the comparison it is on.

    `exa_py` the module is `exa-py` the package. This used to normalise the
    import names and not the keys of DELIBERATELY_OPTIONAL, so an entry for a
    package with an underscore in it never matched - the list looked right and
    the guard failed anyway. Every existing entry happened to be a single word,
    which is why nothing caught it until one wasn't.
    """
    return name.strip().lower().replace("_", "-")


def test_every_import_is_declared_or_a_written_down_decision():
    known = declared() | {normalise(k) for k in DELIBERATELY_OPTIONAL}
    undeclared = {
        name: sorted(where) for name, where in imported().items()
        if normalise(name) not in known
    }
    assert not undeclared, (
        "imported by the shipped code but neither declared in requirements.txt "
        "nor listed as deliberately optional: "
        + "; ".join(f"{n} ({', '.join(w)})" for n, w in sorted(undeclared.items()))
        + ".  Add it to requirements.txt, or to DELIBERATELY_OPTIONAL with the "
        "reason it is safe to be missing - and if it is optional, make sure the "
        "code says so out loud when it is."
    )


def test_the_optional_list_has_not_gone_stale():
    """The other direction. A name left behind after its import is deleted
    reads as a dependency that exists, and quietly widens what the next person
    thinks they are allowed to leave undeclared."""
    live = {normalise(name) for name in imported()}
    stale = sorted(k for k in DELIBERATELY_OPTIONAL if normalise(k) not in live)
    assert not stale, f"no longer imported by anything shipped: {stale}"


def test_numpy_is_declared_because_the_synthesis_path_needs_it():
    """The specific regression. `pcm_from_float` runs with chatterbox and torch
    stubbed out - which is exactly the environment CI installs - so numpy
    cannot be filed under the deep-learning stack."""
    assert "numpy" in declared(), (
        "numpy is imported by tts.pcm_from_float on a path that runs without "
        "torch; leaving it undeclared is what turned CI red"
    )
    assert "numpy" not in DELIBERATELY_OPTIONAL


def test_the_engine_contract_really_does_run_without_the_gpu_stack():
    """The reason the fix is a declaration and not a skip.

    Making the chatterbox tests skip when numpy is missing would have turned CI
    green while removing the only coverage the production voice has on a
    machine that cannot run it. Nothing in the suite may opt out that way.
    """
    for name in ("test_engine_contract.py", "test_chatterbox_engine.py"):
        source = (ROOT / "tests" / name).read_text()
        assert "importorskip" not in source, (
            f"{name} skips itself when a package is missing; CI is the only "
            "place that runs a clean install, so that is the one environment "
            "where the coverage would silently disappear"
        )
