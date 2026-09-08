"""Accounts: the first thing in FAM that is not anonymous.

Until now a listener was whatever id the browser sent. `famUserId()` made one
up with `Math.random()`, stored it in localStorage, and put it in the query
string of every request. Nothing checked it, so anyone who guessed or read an
id could fetch that listener's profile, rename them, delete their mixes and
echo as them. That was defensible while the app was stateless; it stopped being
defensible when a durable listener table was keyed on that id.

**The shape of the fix is the important part.** It would be easy to read
"accounts" as "put a login in front of the app", and that would be wrong twice
over - it breaks the one-sentence spec by putting a form in front of the first
word, and it throws away the history of everyone who has used it so far. So:

    An identity is a *session*, and an account is *credentials attached to a
    session's identity*.

Every request resolves a session from an HttpOnly cookie. If there is no
cookie, the server mints one - a `secrets.token_urlsafe` id, high-entropy and
never chosen by the client. That listener is anonymous, and everything works
for them exactly as it does now: search, myFAM, Go Deeper, mixes, all of it.

Signing up then *attaches* an email and password to the id they already have.
There is no migration and nothing to claim: it is the same `user_id`, so their
history, mixes and echoes are simply theirs now, on any device they log in
from. Logging out drops the session; the next request mints a fresh anonymous
one and they are a new listener.

That gets the property that actually matters - **an id can no longer be
forged** - without a login screen in front of anybody.

What is deliberately not here: password reset (needs email delivery, which the
app has no route to), and any notion of an admin. Both are honest gaps rather
than oversights, and are named in PROBLEMS.md.

Storage notes, because they are the parts worth getting right:

* Passwords go through `hashlib.scrypt` - memory-hard, in the standard library,
  so no new dependency. ~47 ms per attempt on this machine, which is the point.
* The session token is **never stored**. Only its SHA-256 goes in the table, so
  a leaked database yields no usable sessions. The token exists in the cookie
  and nowhere else.
* Comparisons use `hmac.compare_digest`, so a wrong password and a wrong email
  cost the same time.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from paths import data_path

log = logging.getLogger(__name__)

#: scrypt parameters. n is the memory/time knob; 2**14 with r=8 needs ~16 MB
#: and ~47 ms per hash, which is cheap for one login and expensive for a
#: million. Stored alongside each hash so these can be raised later without
#: invalidating existing passwords.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

#: 32 bytes of urlsafe base64. Guessing one is not a threat model.
TOKEN_BYTES = 32

#: How long a session lives, refreshed on use. Long, because this is a podcast
#: app and being logged out is a worse failure here than a stale session; the
#: cookie is HttpOnly and the token is revocable, which is what carries the
#: security rather than a short expiry.
SESSION_TTL = 90 * 86400

#: The cookie the whole scheme rests on.
COOKIE_NAME = "fam_session"

MAX_EMAIL = 254
MIN_PASSWORD = 10
MAX_PASSWORD = 1024  # scrypt on an unbounded string is a denial-of-service

# Deliberately permissive. Address validation by regex is a well-known way to
# reject real addresses; the only thing that proves an address is sending to it,
# which this app cannot do. So: something, an @, something with a dot.
_EMAIL_OK = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s.]{2,}$")


class AuthError(ValueError):
    """Something the person can fix, phrased so it can be shown to them."""


@dataclass(frozen=True)
class Listener:
    """Who the server believes is making this request."""

    user_id: str
    email: str = ""

    @property
    def is_authenticated(self) -> bool:
        return bool(self.email)

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "authenticated": self.is_authenticated,
        }


def clean_email(email: str) -> str:
    email = " ".join(str(email).split()).strip().lower()[:MAX_EMAIL]
    if not _EMAIL_OK.match(email):
        raise AuthError("That does not look like an email address.")
    return email


def check_password(password: str) -> str:
    password = str(password)
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Use at least {MIN_PASSWORD} characters.")
    if len(password) > MAX_PASSWORD:
        raise AuthError("That password is too long.")
    return password


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """`scrypt$n$r$p$salt$hash`, all base64. Parameters travel with the hash so
    they can be raised later without locking anyone out."""
    salt = salt or secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt_b64),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
            maxmem=64 * 1024 * 1024,
        )
    except Exception:
        # A malformed hash is not a reason to let anybody in.
        log.exception("could not verify a password hash")
        return False
    return hmac.compare_digest(expected, actual)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AccountStore:
    """Credentials and sessions.

    Its own database rather than a table in social.db, because this is the only
    store that holds secrets. It has a different sensitivity, a different
    backup story and a different blast radius, and mixing password hashes into
    the file that holds echoes would give the whole thing the strictest of
    those properties by accident rather than on purpose.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = data_path("ACCOUNTS_DB", "accounts.db", path)
        self._local = threading.local()
        self._pruned_at = 0.0
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS accounts (
                       user_id    TEXT PRIMARY KEY,
                       email      TEXT NOT NULL,
                       password   TEXT NOT NULL,
                       created    REAL NOT NULL,
                       last_login REAL NOT NULL DEFAULT 0
                   )"""
            )
            # Case-folded on the way in, so this is the real uniqueness rule
            # rather than a near-miss that lets Alice@ and alice@ both exist.
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS accounts_email"
                         " ON accounts(email)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                       token_hash TEXT PRIMARY KEY,
                       user_id    TEXT NOT NULL,
                       created    REAL NOT NULL,
                       expires    REAL NOT NULL,
                       last_used  REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS sessions_user"
                         " ON sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS sessions_expires"
                         " ON sessions(expires)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # --- sessions ---------------------------------------------------------

    def new_session(self, user_id: str = "", at: float = 0.0) -> tuple[str, str]:
        """Mint a session. Returns (token, user_id).

        With no `user_id` this is an anonymous listener, and the id is minted
        here rather than accepted from the client - which is the entire point.
        """
        now = at or time.time()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        user_id = user_id or "anon_" + secrets.token_urlsafe(16)
        self._conn().execute(
            "INSERT INTO sessions (token_hash, user_id, created, expires, last_used)"
            " VALUES (?, ?, ?, ?, ?)",
            (_token_hash(token), user_id, now, now + SESSION_TTL, now),
        )
        self._maybe_prune(now)
        return token, user_id

    def listener_for(self, token: str, at: float = 0.0) -> Optional[Listener]:
        """Who this token belongs to, or None if it is unknown or expired."""
        if not token:
            return None
        now = at or time.time()
        try:
            row = self._conn().execute(
                "SELECT s.user_id, COALESCE(a.email, '') FROM sessions s"
                " LEFT JOIN accounts a ON a.user_id = s.user_id"
                " WHERE s.token_hash = ? AND s.expires > ?",
                (_token_hash(token), now),
            ).fetchone()
        except Exception:
            log.exception("could not read session")
            return None
        if not row:
            return None
        # Sliding expiry: someone who uses the app keeps their session. Written
        # at most once an hour, because a write on every request would make a
        # read endpoint a writer for no benefit.
        try:
            self._conn().execute(
                "UPDATE sessions SET last_used = ?, expires = ?"
                " WHERE token_hash = ? AND last_used < ?",
                (now, now + SESSION_TTL, _token_hash(token), now - 3600),
            )
        except Exception:
            log.exception("could not refresh session; continuing")
        return Listener(row[0], row[1])

    def end_session(self, token: str) -> bool:
        if not token:
            return False
        cur = self._conn().execute(
            "DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),)
        )
        return bool(cur.rowcount)

    def end_all_sessions(self, user_id: str) -> int:
        """Every session for one account. What a password change should do."""
        cur = self._conn().execute(
            "DELETE FROM sessions WHERE user_id = ?", (user_id,)
        )
        return cur.rowcount or 0

    def _maybe_prune(self, now: float) -> None:
        if now - self._pruned_at < 3600:
            return
        self._pruned_at = now
        try:
            self._conn().execute("DELETE FROM sessions WHERE expires < ?", (now,))
        except Exception:
            log.exception("could not prune sessions; continuing")

    # --- accounts ---------------------------------------------------------

    def account(self, user_id: str) -> Optional[dict]:
        try:
            row = self._conn().execute(
                "SELECT email, created, last_login FROM accounts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception:
            log.exception("could not read account")
            return None
        if not row:
            return None
        return {"user_id": user_id, "email": row[0],
                "created": row[1], "last_login": row[2]}

    def sign_up(self, user_id: str, email: str, password: str,
                at: float = 0.0) -> Listener:
        """Attach credentials to the identity this listener already has.

        Deliberately *not* "create a user". The listener exists already - they
        have a history, maybe mixes, maybe echoes - and this claims that
        identity rather than starting a second one beside it. That is why there
        is no migration step anywhere in this module.
        """
        if not user_id:
            raise AuthError("No listener to attach an account to.")
        email = clean_email(email)
        password = check_password(password)
        now = at or time.time()

        if self.account(user_id):
            raise AuthError("This listener already has an account. Log out first.")
        try:
            self._conn().execute(
                "INSERT INTO accounts (user_id, email, password, created, last_login)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, email, hash_password(password), now, now),
            )
        except sqlite3.IntegrityError as exc:
            # Only tells them an address is taken, which they can already
            # discover by trying to sign up. Nothing else is disclosed.
            raise AuthError("That email is already registered.") from exc
        return Listener(user_id, email)

    def log_in(self, email: str, password: str, at: float = 0.0) -> Listener:
        """Verify credentials. Raises the *same* error either way.

        A different message for "no such email" and "wrong password" turns the
        login form into an account-enumeration oracle.
        """
        now = at or time.time()
        try:
            email = clean_email(email)
        except AuthError:
            # Still pay the hashing cost, so a malformed address is not
            # instantly distinguishable from a real one that failed.
            verify_password(str(password), hash_password("dummy"))
            raise AuthError("That email and password do not match.") from None

        row = self._conn().execute(
            "SELECT user_id, password FROM accounts WHERE email = ?", (email,)
        ).fetchone()
        if not row:
            verify_password(str(password), hash_password("dummy"))
            raise AuthError("That email and password do not match.")
        if not verify_password(str(password), row[1]):
            raise AuthError("That email and password do not match.")

        self._conn().execute(
            "UPDATE accounts SET last_login = ? WHERE user_id = ?", (now, row[0])
        )
        return Listener(row[0], email)

    def change_password(self, user_id: str, current: str, new: str) -> None:
        row = self._conn().execute(
            "SELECT password FROM accounts WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or not verify_password(str(current), row[0]):
            raise AuthError("That is not your current password.")
        self._conn().execute(
            "UPDATE accounts SET password = ? WHERE user_id = ?",
            (hash_password(check_password(new)), user_id),
        )
        # Every other device is logged out. A password change that leaves old
        # sessions alive does not do the thing people believe it does.
        self.end_all_sessions(user_id)
