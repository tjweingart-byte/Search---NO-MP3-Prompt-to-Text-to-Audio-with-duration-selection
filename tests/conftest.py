"""Keep the suite hermetic.

The suite must produce the same result on every machine. Two ways a developer's
environment used to leak in, both of which turned "the code changed" into "this
laptop is different":

1. **`.env` and `~/.fam/env`.** config.py reads both, so that `python app.py`
   finds the key however the server is started. That is right for the app and
   wrong for the tests: a key there flips the app out of demo mode mid-suite -
   a different model, a different cache key, a different code path.

2. **An exported `ANTHROPIC_API_KEY`.** Blocking the files was not enough,
   because anyone who has run the server has the variable in their shell. With
   it set, `/api/script` stops being a stub and makes a real Claude call - and
   `test_generation_is_still_paced` then failed on a real machine and passed in
   CI, because a real call takes longer than the 3-second pacing window it was
   asserting against. The limiter was right; the test was reading the machine.

So the key is removed here rather than worked around per test. Nothing in this
suite may reach Claude: every test that needs a generator brings its own.
"""
import os

os.environ["FAM_IGNORE_DOTENV"] = "1"

# Not `del`: it must go whether or not it was there.
os.environ.pop("ANTHROPIC_API_KEY", None)
