"""Keep the suite hermetic.

config.py now reads .env, so that `python app.py` finds the key however the
server is started. That is right for the app and wrong for the tests: a
developer with a real key in .env would run a different suite from CI - demo
mode off, a different model, a different cache key - and the failure would look
like a code change rather than an environment.
"""
import os

import pytest

os.environ["FAM_IGNORE_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def isolated_accounts(tmp_path, monkeypatch):
    """Every test gets its own sessions and credentials.

    Without this the suite would mint session rows into the real accounts.db in
    the project root - the store holding password hashes, which is the last one
    that should collect debris from a test run.
    """
    import accounts as accounts_mod
    import app as appmod

    # In a subdirectory, not tmp_path itself: a test that asserts no store
    # followed the working directory looks for these filenames in the cwd, and
    # this fixture's own file would otherwise look like the very stray it hunts.
    monkeypatch.setattr(
        appmod,
        "ACCOUNTS",
        accounts_mod.AccountStore(str(tmp_path / "auth" / "accounts.db")),
    )
