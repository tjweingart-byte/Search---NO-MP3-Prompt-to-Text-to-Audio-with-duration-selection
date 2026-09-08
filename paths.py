"""Where the databases live.

Every store used to default to a bare filename - `myfam.db`, `social.db` and
so on. A bare filename is resolved against the **current working directory**,
which is not a property of the app at all: it is a property of wherever the
person who started the process happened to be standing.

That failed in the way this project dislikes most - silently, and in the
direction of looking fine. Starting the server from a different directory did
not raise anything. SQLite simply created a second, empty set of files, so the
app came up with a cold script cache (every episode paid ~$0.03 again), an
empty feed, no echoes and no mixes, and nothing anywhere said so. The same trap
caught `tools/seed_demo.py`, which writes the history the browse surfaces need:
seed from one directory, serve from another, and Explore stays empty however
much you tap it - which reads exactly like a broken feature.

So a path is now resolved in one place, from `__file__` rather than the cwd:

* **No env var** - the file sits beside this module, in the project root. Same
  location however the process was started.
* **An absolute env var** - used exactly as given. This is what a deployment
  does, and it is the only form a deployment should use: the Dockerfile points
  all five at the mounted `/data` disk so they survive a redeploy.
* **A relative env var** - resolved against the project root too, never the
  cwd, and it says so in the log. Reinterpreting someone's input quietly would
  reintroduce the ambiguity this module exists to remove.

There is no fallback that invents a location, so there is no way to end up
reading one file and writing another.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

#: The directory this file sits in. Derived from `__file__`, so it names the
#: same place no matter where the process was started from - which is the
#: whole point, and the reason this is not `Path.cwd()`.
PROJECT_ROOT = Path(__file__).resolve().parent


def data_path(env_var: str, filename: str) -> str:
    """Absolute path for one database, from `env_var` or the project root.

    Read at call time rather than at import, so a test that sets the variable
    and rebuilds a store gets what it set.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return str(PROJECT_ROOT / filename)

    given = Path(raw).expanduser()
    if given.is_absolute():
        return str(given)

    resolved = PROJECT_ROOT / given
    # Announced rather than silent: a relative override is ambiguous, and the
    # ambiguity is the bug. Saying which file was opened costs one log line
    # and saves the afternoon spent wondering why the cache is cold.
    log.warning(
        "%s=%r is relative; resolving against the project root -> %s "
        "(set an absolute path to choose the location yourself)",
        env_var, raw, resolved,
    )
    return str(resolved)
