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

# Tiers live in one place. `metering.PLANS` is the same tuple, re-exported
# there because the cost report is what splits on it - but what a plan
# *allows* is this module's business, so it asks entitlements directly.
import entitlements
from metering import PLANS
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
MAX_PHONE = 20          # E.164 is at most 15 digits; the rest is the + and slack
MAX_DISPLAY_NAME = 60
MIN_PASSWORD = 10
MAX_PASSWORD = 1024  # scrypt on an unbounded string is a denial-of-service

# Deliberately permissive. Address validation by regex is a well-known way to
# reject real addresses; the only thing that proves an address is sending to it,
# which this app cannot do. So: something, an @, something with a dot.
_EMAIL_OK = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s.]{2,}$")

# E.164 and nothing else: a leading +, a non-zero country digit, then digits.
# Strict where the email rule is permissive, and for the opposite reason - an
# address has one canonical form and a phone number has a dozen, so the only
# way two people cannot end up owning "the same" number in different notations
# is to store exactly one form. Everything else is normalised into it or
# refused, rather than being stored as typed and compared hopefully.
_PHONE_OK = re.compile(r"^\+[1-9]\d{7,14}$")

#: How an identity was established. "email" and "phone" are credentials this
#: server holds; "google" and "apple" are assertions it verifies and does not
#: store a secret for.
PROVIDERS = ("email", "phone", "google", "apple")


class AuthError(ValueError):
    """Something the person can fix, phrased so it can be shown to them."""


@dataclass(frozen=True)
class Listener:
    """Who the server believes is making this request.

    Carries the tier because every generation asks for it, and resolving the
    session already touches the accounts table - a second query per episode to
    answer a question the first one could have would put a database round trip
    on the path the whole product is judged by.
    """

    user_id: str
    email: str = ""
    tier: str = "free"
    display_name: str = ""
    phone: str = ""
    #: Whether an account row exists for this id. Its own field rather than
    #: something inferred from the others, because an account created with Sign
    #: in with Apple may legitimately have no email and no phone at all - Apple
    #: sends an address on the first authorization only, and it may be a relay.
    #: Inferring from `email` called such a listener anonymous and offered them
    #: a sign-up screen on every open.
    has_account: bool = False

    @property
    def is_authenticated(self) -> bool:
        """Is there an account behind this session, by any route?"""
        return bool(self.has_account or self.email or self.phone)

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "phone": self.phone,
            "display_name": self.display_name,
            "tier": self.tier,
            "authenticated": self.is_authenticated,
            "has_account": self.has_account,
        }


def clean_email(email: str) -> str:
    email = " ".join(str(email).split()).strip().lower()[:MAX_EMAIL]
    if not _EMAIL_OK.match(email):
        raise AuthError("That does not look like an email address.")
    return email


def clean_phone(phone: str) -> str:
    """Any way someone types a number, into the one form that is stored.

    Spaces, dashes, brackets and dots go; a leading `00` becomes `+`, which is
    how most of the world dials internationally. What is deliberately *not*
    done is guessing a country code for a bare national number: "(555) 123-4567"
    could be a dozen countries, and picking one silently would let two people
    own the same account. It is refused, with a message that says what to do.
    """
    raw = " ".join(str(phone).split()).strip()[:MAX_PHONE * 2]
    cleaned = re.sub(r"[\s().\-]", "", raw)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if not cleaned.startswith("+"):
        raise AuthError("Include the country code, starting with + "
                        "(for example +1 for the US).")
    if not _PHONE_OK.match(cleaned):
        raise AuthError("That does not look like a phone number.")
    return cleaned


def clean_display_name(name: str) -> str:
    """Whitespace-collapsed and bounded. Not otherwise policed: a name is
    whatever somebody says it is, and a filter here is a filter that rejects
    real names."""
    return " ".join(str(name or "").split())[:MAX_DISPLAY_NAME]


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
            # Added after the table shipped, so an existing accounts file is
            # widened rather than recreated - the same trade topics.py makes.
            #
            # Nothing in this app sets it to "paid": there is no payment route
            # and nothing is gated on having an account (CLAUDE.md's open
            # question about what an account should entitle you to is still
            # open). It exists so that the day something does take payment, the
            # cost split by plan is available from that day forward instead of
            # being backfilled out of a log that never recorded it.
            for ddl in (
                "ALTER TABLE accounts ADD COLUMN plan TEXT NOT NULL DEFAULT 'free'",
                # Sign in with Apple can create an account with no address, and
                # phone sign-up creates one with no address on purpose. Both
                # need somewhere for a name to live that is not the email.
                "ALTER TABLE accounts ADD COLUMN display_name TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE accounts ADD COLUMN phone TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # already there

            # The email index shipped as a plain UNIQUE, which was right when
            # every account had an address. It is not any more: an account can
            # now be phone-only or Apple-only, and SQLite treats two empty
            # strings as equal - so the second such account would collide with
            # the first on a value that means "no email".
            #
            # Replaced with partial indexes that only constrain rows which
            # actually hold a value. Dropped by name and recreated, because an
            # existing accounts.db already has the old one.
            conn.execute("DROP INDEX IF EXISTS accounts_email")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS accounts_email"
                         " ON accounts(email) WHERE email != ''")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS accounts_phone"
                         " ON accounts(phone) WHERE phone != ''")

            # How an account can be reached, one row per route. Separate from
            # `accounts` because it is many-to-one: the same person may sign in
            # with Apple on their phone and a password on the web, and both
            # must land on one listener rather than two.
            #
            # Keyed on (provider, subject) rather than on email. Apple sends an
            # address on the first authorization only, so an email-keyed table
            # would create a second account on the second sign-in - the single
            # most common way this gets built wrong.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                       provider   TEXT NOT NULL,
                       subject    TEXT NOT NULL,
                       user_id    TEXT NOT NULL,
                       email      TEXT NOT NULL DEFAULT '',
                       created    REAL NOT NULL,
                       last_login REAL NOT NULL DEFAULT 0,
                       PRIMARY KEY (provider, subject)
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS identities_user"
                         " ON identities(user_id)")

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
                "SELECT s.user_id, COALESCE(a.email, ''), COALESCE(a.plan, 'free'),"
                "       COALESCE(a.display_name, ''), COALESCE(a.phone, ''),"
                "       a.user_id IS NOT NULL"
                " FROM sessions s"
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
        return Listener(row[0], row[1], entitlements.normalise(row[2]),
                        row[3], row[4], bool(row[5]))

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
                "SELECT email, created, last_login, plan, display_name, phone"
                " FROM accounts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception:
            log.exception("could not read account")
            return None
        if not row:
            return None
        return {"user_id": user_id, "email": row[0],
                "created": row[1], "last_login": row[2],
                "plan": entitlements.normalise(row[3]),
                "display_name": row[4], "phone": row[5]}

    def plan_for(self, user_id: str) -> str:
        """Which plan to stamp on this listener's usage rows.

        **An anonymous listener is "free", not "unknown".** They are a real
        listener costing real money, and a metering report that split them into
        a third bucket would answer "what does a free user cost" with a number
        that excluded most of them.
        """
        account = self.account(user_id) if user_id else None
        return entitlements.normalise((account or {}).get("plan") or "free")

    def set_plan(self, user_id: str, plan: str) -> str:
        """Move an account between plans. Requires an account: a plan is a
        billing relationship, and there is nobody to bill without one."""
        # Compared against the raw names, not through `normalise` - which is
        # total and would turn a typo into "free" and report success. A caller
        # setting a plan is making a billing decision and has to be told it did
        # not land.
        if plan not in PLANS and plan not in entitlements.LEGACY_TIERS:
            raise AuthError(f"Unknown plan {plan!r}. One of: {', '.join(PLANS)}.")
        plan = entitlements.normalise(plan)
        if not self.account(user_id):
            raise AuthError("That listener has no account, so it has no plan.")
        self._conn().execute("UPDATE accounts SET plan = ? WHERE user_id = ?",
                             (plan, user_id))
        return plan

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
        self._link(user_id, "email", email, email, now)
        return Listener(user_id, email, has_account=True)

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
        return self.listener_of(row[0])

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

    def listener_of(self, user_id: str) -> Listener:
        """The Listener for an id, read straight from the account row.

        Used where a session has not been minted yet - the moment after a login
        succeeds, mostly. Everything else resolves through `listener_for`,
        which starts from a token; this one starts from an id that has already
        been proven.
        """
        account = self.account(user_id)
        if not account:
            return Listener(user_id)
        return Listener(user_id, account["email"], account["plan"],
                        account["display_name"], account["phone"], True)

    # --- identities -------------------------------------------------------

    def _link(self, user_id: str, provider: str, subject: str, email: str,
              at: float) -> None:
        """Record that this route reaches this listener. Idempotent."""
        self._conn().execute(
            "INSERT INTO identities (provider, subject, user_id, email,"
            " created, last_login) VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (provider, subject) DO UPDATE SET"
            " last_login = excluded.last_login,"
            # Only fill an email in, never blank one out: Apple sends an
            # address on the first authorization and nothing afterwards, so
            # overwriting on every sign-in would erase it on the second one.
            " email = CASE WHEN excluded.email != '' THEN excluded.email"
            "              ELSE identities.email END",
            (provider, subject, user_id, email or "", at, at),
        )

    def identities_for(self, user_id: str) -> list[dict]:
        """Every way into this account. What the settings screen lists, and
        what `unlink_identity` refuses to empty."""
        try:
            rows = self._conn().execute(
                "SELECT provider, subject, email, created, last_login"
                " FROM identities WHERE user_id = ? ORDER BY created",
                (user_id,),
            ).fetchall()
        except Exception:
            log.exception("could not read identities")
            return []
        return [{"provider": r[0],
                 # Never returned in full. A Google or Apple subject is a
                 # stable per-person identifier, and there is no reason for it
                 # to leave the server - it is only here so the screen can tell
                 # two Google links apart.
                 "subject_hint": (r[1][-6:] if r[0] in ("google", "apple") else r[1]),
                 "email": r[2], "created": r[3], "last_login": r[4]}
                for r in rows]

    def user_for_identity(self, provider: str, subject: str) -> str:
        row = self._conn().execute(
            "SELECT user_id FROM identities WHERE provider = ? AND subject = ?",
            (provider, subject),
        ).fetchone()
        return row[0] if row else ""

    def sign_in_with(self, provider: str, subject: str, *, email: str = "",
                     display_name: str = "", current_user_id: str = "",
                     at: float = 0.0) -> tuple[Listener, bool]:
        """Sign in - or sign up - with a verified provider identity.

        Returns (listener, is_new). The caller has already *verified* the
        token; this module never sees one, which keeps the crypto in `oauth.py`
        and the storage here.

        Three cases, and the third is the one with a decision in it:

        1. **Known identity** - log in as that account, whoever is holding the
           current session.
        2. **New identity, current listener already has an account** - link it.
           This is "add Google to my account", and it is why one person can
           have a password and two providers without three sets of history.
        3. **New identity, anonymous listener** - attach an account to the id
           they already have, exactly as `sign_up` does. Their listening so far
           is theirs, rather than being stranded on an id they just left.

        The email is stored **only if no other account holds it**. Linking on a
        matching address would let anyone who can get a provider to assert an
        address take over the account that owns it, and providers differ on how
        hard that is. So the address is a detail here, never a key: the account
        is reached by `(provider, subject)`, which is the only field either
        provider guarantees.
        """
        if provider not in PROVIDERS:
            raise AuthError(f"Unknown sign-in provider {provider!r}.")
        subject = str(subject or "").strip()
        if not subject:
            raise AuthError("That sign-in carried no identity.")
        now = at or time.time()
        email = clean_email(email) if email else ""
        display_name = clean_display_name(display_name)

        existing = self.user_for_identity(provider, subject)
        if existing:
            self._link(existing, provider, subject, email, now)
            self._conn().execute(
                "UPDATE accounts SET last_login = ? WHERE user_id = ?",
                (now, existing))
            return self.listener_of(existing), False

        if not current_user_id:
            raise AuthError("No listener to attach an account to.")

        if not self.account(current_user_id):
            taken = bool(email) and bool(self._conn().execute(
                "SELECT 1 FROM accounts WHERE email = ?", (email,)).fetchone())
            self._conn().execute(
                "INSERT INTO accounts (user_id, email, password, created,"
                " last_login, display_name) VALUES (?, ?, '', ?, ?, ?)",
                (current_user_id, "" if taken else email, now, now, display_name),
            )
        self._link(current_user_id, provider, subject, email, now)
        return self.listener_of(current_user_id), True

    def unlink_identity(self, user_id: str, provider: str) -> int:
        """Remove a sign-in route, unless it is the last way in.

        The refusal is the point. An account whose only identity is Apple, with
        no password set, becomes unreachable the moment that link is dropped -
        and it would look like it worked.
        """
        rows = self.identities_for(user_id)
        has_password = bool((self._conn().execute(
            "SELECT password FROM accounts WHERE user_id = ?",
            (user_id,)).fetchone() or [""])[0])
        remaining = [r for r in rows if r["provider"] != provider]
        if not remaining and not has_password:
            raise AuthError(
                "That is the only way into this account. Add another sign-in "
                "method, or set a password, before removing it.")
        cur = self._conn().execute(
            "DELETE FROM identities WHERE user_id = ? AND provider = ?",
            (user_id, provider))
        return cur.rowcount or 0

    # --- phone ------------------------------------------------------------

    def sign_up_phone(self, user_id: str, phone: str, password: str,
                      at: float = 0.0) -> Listener:
        """The same attach-to-the-id-they-have as `sign_up`, keyed on a number.

        **The number is not verified.** Nothing here sends an SMS, so this
        proves possession of a password and no more - a phone number is an
        identifier, not a second factor, until something delivers a code to it.
        `ACCOUNTS.md` says so out loud, and the sign-up screen must too, on the
        same reasoning that made the missing password reset a stated gap rather
        than a surprise.
        """
        if not user_id:
            raise AuthError("No listener to attach an account to.")
        phone = clean_phone(phone)
        password = check_password(password)
        now = at or time.time()
        if self.account(user_id):
            raise AuthError("This listener already has an account. Log out first.")
        try:
            self._conn().execute(
                "INSERT INTO accounts (user_id, email, password, created,"
                " last_login, phone) VALUES (?, '', ?, ?, ?, ?)",
                (user_id, hash_password(password), now, now, phone),
            )
        except sqlite3.IntegrityError as exc:
            raise AuthError("That phone number is already registered.") from exc
        self._link(user_id, "phone", phone, "", now)
        return self.listener_of(user_id)

    def log_in_phone(self, phone: str, password: str, at: float = 0.0) -> Listener:
        """As `log_in`, with the same single error for every failure."""
        now = at or time.time()
        try:
            phone = clean_phone(phone)
        except AuthError:
            verify_password(str(password), hash_password("dummy"))
            raise AuthError("That number and password do not match.") from None

        row = self._conn().execute(
            "SELECT user_id, password FROM accounts WHERE phone = ?", (phone,)
        ).fetchone()
        if not row or not row[1] or not verify_password(str(password), row[1]):
            # `not row[1]` is an account with no password - one created with
            # Google or Apple. It must cost the same time and give the same
            # message as a wrong password, or the response distinguishes "this
            # account exists and uses Apple" for anyone who asks.
            verify_password(str(password), hash_password("dummy"))
            raise AuthError("That number and password do not match.")
        self._conn().execute(
            "UPDATE accounts SET last_login = ? WHERE user_id = ?", (now, row[0]))
        return self.listener_of(row[0])

    # --- settings ---------------------------------------------------------

    def update_profile(self, user_id: str, *, display_name: Optional[str] = None,
                       email: Optional[str] = None,
                       phone: Optional[str] = None) -> dict:
        """Change what the account says about itself.

        Only the fields that were passed. `None` means "leave it alone" and an
        empty string means "remove it", which are different requests and would
        be the same one if absence were spelled `""`.
        """
        account = self.account(user_id)
        if not account:
            raise AuthError("That listener has no account.")

        sets, values = [], []
        if display_name is not None:
            sets.append("display_name = ?")
            values.append(clean_display_name(display_name))
        if email is not None:
            sets.append("email = ?")
            values.append(clean_email(email) if email else "")
        if phone is not None:
            sets.append("phone = ?")
            values.append(clean_phone(phone) if phone else "")
        if not sets:
            return account

        # Removing the last way in is refused here for the same reason
        # `unlink_identity` refuses it: an account nobody can reach is a
        # deletion that does not say it deleted anything.
        wanted_email = values[sets.index("email = ?")] if "email = ?" in sets else account["email"]
        wanted_phone = values[sets.index("phone = ?")] if "phone = ?" in sets else account["phone"]
        others = [i for i in self.identities_for(user_id)
                  if i["provider"] in ("google", "apple")]
        if not wanted_email and not wanted_phone and not others:
            raise AuthError("Removing that would leave no way to sign in.")

        values.append(user_id)
        try:
            self._conn().execute(
                f"UPDATE accounts SET {', '.join(sets)} WHERE user_id = ?",
                tuple(values))
        except sqlite3.IntegrityError as exc:
            raise AuthError("That email or phone number is already registered.") from exc

        # Keep the identity rows in step, so "how do I get back in" and "what
        # does my profile say" cannot drift apart.
        now = time.time()
        if email is not None and wanted_email:
            self._conn().execute(
                "DELETE FROM identities WHERE user_id = ? AND provider = 'email'",
                (user_id,))
            self._link(user_id, "email", wanted_email, wanted_email, now)
        if phone is not None and wanted_phone:
            self._conn().execute(
                "DELETE FROM identities WHERE user_id = ? AND provider = 'phone'",
                (user_id,))
            self._link(user_id, "phone", wanted_phone, "", now)
        return self.account(user_id)

    def set_password(self, user_id: str, new: str) -> None:
        """Set a password where there was none - the Google-or-Apple account
        adding a second way in. Separate from `change_password`, which requires
        the current one and would be impossible to satisfy here."""
        row = self._conn().execute(
            "SELECT password FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            raise AuthError("That listener has no account.")
        if row[0]:
            raise AuthError("This account already has a password. Change it instead.")
        self._conn().execute(
            "UPDATE accounts SET password = ? WHERE user_id = ?",
            (hash_password(check_password(new)), user_id))

    # --- sessions the listener can see ------------------------------------

    def sessions_for(self, user_id: str, current_token: str = "",
                     at: float = 0.0) -> list[dict]:
        """Where this account is signed in. No token is returned, only its
        age - a list of live credentials is not something to hand back over
        the wire, even to their owner."""
        now = at or time.time()
        try:
            rows = self._conn().execute(
                "SELECT token_hash, created, last_used, expires FROM sessions"
                " WHERE user_id = ? AND expires > ? ORDER BY last_used DESC",
                (user_id, now),
            ).fetchall()
        except Exception:
            log.exception("could not list sessions")
            return []
        current = _token_hash(current_token) if current_token else ""
        return [{"created": r[1], "last_used": r[2], "expires": r[3],
                 "current": bool(current) and r[0] == current}
                for r in rows]

    def end_other_sessions(self, user_id: str, keep_token: str) -> int:
        """"Sign out everywhere else." Keeps the one asking, which is the
        difference between this and `end_all_sessions` and the reason both
        exist: a password change should log this device out too."""
        cur = self._conn().execute(
            "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
            (user_id, _token_hash(keep_token) if keep_token else ""))
        return cur.rowcount or 0

    # --- deletion ---------------------------------------------------------

    def delete_account(self, user_id: str) -> dict:
        """Erase the credentials, the identities and every session.

        This is the *credential* half of deleting a listener. The rest of what
        FAM knows about them lives in the other stores, and `app.py` walks them
        all - see `erase_listener` there, which is the whole operation and the
        one an endpoint should call.

        Deliberately unconditional: it does not ask for a password. The caller
        has already proved it is them by holding the session, and an account
        created with Sign in with Apple has no password to ask for - a
        confirmation step that only some accounts can satisfy is a deletion
        route that only some accounts have.
        """
        account = self.account(user_id) or {}
        identities = len(self.identities_for(user_id))
        sessions = self.end_all_sessions(user_id)
        self._conn().execute("DELETE FROM identities WHERE user_id = ?", (user_id,))
        cur = self._conn().execute("DELETE FROM accounts WHERE user_id = ?",
                                   (user_id,))
        return {"account": bool(cur.rowcount), "identities": identities,
                "sessions": sessions, "email": account.get("email", "")}
