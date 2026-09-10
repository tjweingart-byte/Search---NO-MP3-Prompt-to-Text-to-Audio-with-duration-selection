"""Sign in with Google, sign in with Apple: verifying what the app was handed.

Both providers work the same way from a native app's point of view, and it is
the *simple* way rather than the redirect dance:

1. The app asks the platform SDK to sign the person in.
2. The SDK hands the app a signed **identity token** - a JWT.
3. The app posts that token here.
4. This module checks the signature against the provider's published keys and
   returns who it says they are.

There is no authorization-code exchange, no client secret, and no redirect URI,
because a native app does not need one. That removes the single most common
way this gets built wrong - a client secret shipped inside the app.

## The rule that carries all the security

**A token is either verified or refused. There is no third branch.** Not
"decode it without checking the signature if the library is missing", not "trust
it in development". A JWT is a plain base64 envelope: anyone can write one that
says they are anybody, so an unverified token is not a weak credential, it is no
credential at all. This project's settled rule is that every fallback must
announce itself; here the only acceptable fallback is a refusal, because the
quiet one hands over every account on the server.

So when the verifying library is not installed, or the provider has no audience
configured, sign-in with that provider is **unavailable** and says so - at
startup, on `/api/health`, and to the app that tried. It never degrades.

## What is checked, and why each one

* **Signature**, against the provider's current JWKS, fetched over HTTPS and
  cached. This is the whole point.
* **Issuer** must be the provider's own. Without it a token from any issuer
  whose key happens to be in the key set would pass.
* **Audience** must be one of *our* client ids. Without it, a token minted for
  a completely different app - which its developers can read - logs in here.
  This is the check that is most often skipped and it is the most dangerous one
  to skip.
* **Expiry**, handled by the library.
* **Nonce**, when the app sends one. It is what stops a token captured from one
  sign-in being replayed into another.

## What comes back, and what does not

The **subject** (`sub`) is the identity. It is stable per provider per person,
and it is the only field that is. The email is a bonus:

* Apple sends it on the **first** authorization only. Every later sign-in for
  the same person carries no email at all, so an account keyed on email would
  create a second account on the second sign-in.
* Apple's email may be a **private relay** address. It works as an address but
  it is a forwarding alias the person can switch off, so it must never be
  presented as "your email address" without qualification, and password reset
  to it is not a guarantee of anything.

Which is why `accounts.py` keys these on `(provider, subject)` and treats the
email as a display detail.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

#: The one line that makes Google and Apple sign-in work on a machine.
INSTALL_HINT = "pip install -r requirements-oauth.txt"

PROVIDERS: tuple[str, ...] = ("google", "apple")


class OAuthUnavailable(RuntimeError):
    """This server cannot verify tokens for this provider.

    An operator's problem, not the person's: a missing library or an
    unconfigured audience. Separate from `OAuthError` because the two want
    different status codes and completely different messages - one is "we are
    broken", the other is "that token is not good".
    """


class OAuthError(ValueError):
    """The token was checked and refused. Safe to show to the person."""


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    #: More than one because Google mints both forms and has for years.
    issuers: tuple[str, ...]
    jwks_uri: str
    #: Which setting names the audiences we accept.
    setting: str


SPECS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        name="google",
        issuers=("https://accounts.google.com", "accounts.google.com"),
        jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
        setting="google_client_ids",
    ),
    "apple": ProviderSpec(
        name="apple",
        issuers=("https://appleid.apple.com",),
        jwks_uri="https://appleid.apple.com/auth/keys",
        setting="apple_client_ids",
    ),
}


@dataclass(frozen=True)
class VerifiedIdentity:
    """Who a verified token says this is."""

    provider: str
    #: Stable per provider per person. The account key, and the only field that
    #: can be one.
    subject: str
    email: str = ""
    email_verified: bool = False
    name: str = ""

    @property
    def is_private_relay(self) -> bool:
        """Apple's forwarding alias. True means the address works but is not
        the person's own, and anything that promises to reach them - password
        reset, a receipt - should say so."""
        return self.email.endswith("@privaterelay.appleid.com")

    def as_dict(self) -> dict:
        return {"provider": self.provider, "subject": self.subject,
                "email": self.email, "email_verified": self.email_verified,
                "name": self.name, "private_relay": self.is_private_relay}


def audiences(provider: str) -> tuple[str, ...]:
    """Client ids this server accepts for a provider, from settings."""
    spec = SPECS.get(provider)
    if spec is None:
        return ()
    from config import settings
    raw = getattr(settings, spec.setting, "") or ""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _library():
    """PyJWT, or a refusal naming the fix.

    Imported here rather than at module scope so that importing `oauth` costs
    nothing on a machine that has no need of it - the app imports this module
    unconditionally to report on it.
    """
    try:
        import jwt  # noqa: PLC0415 - deliberately lazy
        from jwt import PyJWKClient  # noqa: PLC0415
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - see below
        # `BaseException`, not `Exception`, and it is not defensive padding.
        # PyJWT's crypto comes from `cryptography`, which is a Rust extension:
        # a version mismatch between the two makes the import raise
        # `pyo3_runtime.PanicException`, which derives from BaseException and
        # sails straight through an `except Exception`. That is a real state a
        # machine gets into - this build container is in it - and the result
        # was a 500 from a Rust panic rather than the honest "sign-in is
        # unavailable, here is the pip line" this whole module is built to give.
        #
        # A partially installed crypto stack is exactly the case that must not
        # become an exception escaping into a request handler, because the one
        # thing worse than sign-in being unavailable is nobody being able to
        # tell that is what happened.
        raise OAuthUnavailable(
            f"This server cannot verify {'/'.join(PROVIDERS)} sign-in: PyJWT is "
            f"not installed, or its crypto backend is broken "
            f"({type(exc).__name__}). {INSTALL_HINT}"
        ) from exc
    return jwt, PyJWKClient


#: One JWKS client per provider. They cache the fetched keys and refresh when
#: a token names a key they have not seen, which is what makes provider key
#: rotation a non-event rather than an outage.
_JWKS: dict[str, object] = {}


def _jwks_client(spec: ProviderSpec):
    client = _JWKS.get(spec.name)
    if client is None:
        _jwt, PyJWKClient = _library()
        client = PyJWKClient(spec.jwks_uri, cache_keys=True)
        _JWKS[spec.name] = client
    return client


def available(provider: str) -> tuple[bool, str]:
    """Can this server complete a sign-in with this provider right now?

    Returns (ready, reason). The reason is written to be read by whoever has to
    fix it, which is why it names the variable and the pip line rather than
    saying "not configured".
    """
    spec = SPECS.get(provider)
    if spec is None:
        return False, f"Unknown provider {provider!r}."
    if not audiences(provider):
        return False, (
            f"{spec.setting.upper()} is empty, so no client id would be "
            f"accepted. Set it to the app's bundle id (and any web client id), "
            f"comma separated."
        )
    try:
        _library()
    except OAuthUnavailable as exc:
        return False, str(exc)
    return True, ""


def report() -> dict:
    """What `/api/health` says about sign-in. Deliberately reports readiness
    per provider rather than one boolean: "sign-in works" is not a fact, and a
    server with Google configured and Apple not is the normal case."""
    out = {}
    for provider in PROVIDERS:
        ready, reason = available(provider)
        out[provider] = {
            "ready": ready,
            "audiences": len(audiences(provider)),
            "reason": reason,
        }
    return out


def _matches_nonce(claim: str, expected: str) -> bool:
    """Compare the token's nonce against what the client says it sent.

    Both forms are accepted because the two providers differ: Google echoes the
    nonce as given, while Sign in with Apple is documented to be sent a SHA-256
    of it, so the token carries the hash. Accepting either means one client
    contract - "send us the raw nonce you used" - rather than a per-provider
    rule the app has to remember.
    """
    if not claim or not expected:
        return False
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    return hmac.compare_digest(claim, expected) or hmac.compare_digest(claim, digest)


def verify(provider: str, id_token: str, nonce: str = "") -> VerifiedIdentity:
    """Check a provider identity token and say who it is.

    Raises `OAuthUnavailable` if this server cannot check it at all, and
    `OAuthError` if it checked and refused. Never returns an unverified
    identity - see the module docstring.
    """
    spec = SPECS.get(provider)
    if spec is None:
        raise OAuthUnavailable(f"Unknown provider {provider!r}.")
    # The client's side of the bargain is checked first, so that "you sent no
    # token" is the answer on every machine rather than being masked by this
    # server's configuration on some of them. An empty token is a bug in the
    # caller whether or not sign-in is available here.
    if not id_token or not isinstance(id_token, str):
        raise OAuthError("No identity token was sent.")
    ready, reason = available(provider)
    if not ready:
        raise OAuthUnavailable(reason)

    jwt, _client_cls = _library()
    try:
        signing_key = _jwks_client(spec).get_signing_key_from_jwt(id_token)
    except OAuthUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a fetch failure and a bad kid look alike
        # Deliberately *not* an OAuthError. A JWKS fetch that failed is this
        # server being unable to check, and calling that "your token is bad"
        # would send someone off to fix a phone that is working.
        raise OAuthUnavailable(
            f"Could not fetch {provider}'s signing keys: {type(exc).__name__}. "
            f"The provider may be unreachable from this server."
        ) from exc

    try:
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=list(audiences(provider)),
            issuer=list(spec.issuers),
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except Exception as exc:  # noqa: BLE001 - the library raises a family of these
        # One message for every failure. A response that distinguished "expired"
        # from "wrong audience" from "bad signature" is a probe for finding out
        # how this server is configured.
        log.info("refused a %s identity token: %s", provider, type(exc).__name__)
        raise OAuthError("That sign-in could not be verified. Try again.") from exc

    subject = str(claims.get("sub") or "")
    if not subject:
        raise OAuthError("That sign-in could not be verified. Try again.")
    if nonce and not _matches_nonce(str(claims.get("nonce") or ""), nonce):
        log.info("refused a %s identity token: nonce mismatch", provider)
        raise OAuthError("That sign-in could not be verified. Try again.")

    # Apple sends this as the string "true"; Google sends a real boolean. Both
    # have to mean the same thing here, and `bool("false")` is True - which is
    # the kind of quiet wrong answer that would mark an unverified address
    # verified.
    raw_verified = claims.get("email_verified", False)
    verified = (raw_verified is True or
                (isinstance(raw_verified, str) and raw_verified.lower() == "true"))

    return VerifiedIdentity(
        provider=provider,
        subject=subject,
        email=str(claims.get("email") or "").strip().lower(),
        email_verified=verified,
        name=str(claims.get("name") or "").strip()[:120],
    )
