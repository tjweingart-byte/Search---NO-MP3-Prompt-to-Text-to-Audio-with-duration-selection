"""Keep the suite hermetic.

config.py now reads .env, so that `python app.py` finds the key however the
server is started. That is right for the app and wrong for the tests: a
developer with a real key in .env would run a different suite from CI - demo
mode off, a different model, a different cache key - and the failure would look
like a code change rather than an environment.
"""
import os

os.environ["FAM_IGNORE_DOTENV"] = "1"
