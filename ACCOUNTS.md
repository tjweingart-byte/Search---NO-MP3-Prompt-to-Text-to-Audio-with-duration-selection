# Accounts, tiers and the public API

What an identity is, how somebody gets one, what each tier allows, and how a
client that is not this server's own web page talks to it. `accounts.py`,
`entitlements.py`, `quotas.py` and `oauth.py` are the code; this is the why.

Read `CLAUDE.md` first for the constraints this sits inside - particularly
**an account gates what is kept, never what is heard**, which decides more of
the shape below than anything else here.

## The one idea

> An **identity** is a session. An **account** is credentials attached to one.
> A **tier** is what that account is allowed to spend.

Every request resolves a session. If there is no session the server mints one,
and that listener is anonymous and can do everything: search, myFAM, DailyFAM,
Explore, Go Deeper, the whole audio path. Signing up *attaches* a way of
proving who they are to the id they already have, so nothing is migrated and
nothing is claimed - their history was already theirs.

That is why there is no login screen in front of the product, and it is why
the tier system below applies to anonymous listeners too. They are on `free`.

## Four ways in, one account

| Provider | What proves it | Verified? |
|---|---|---|
| `email` | password (scrypt) | the address is **not** verified - nothing delivers mail |
| `phone` | password (scrypt) | the number is **not** verified - nothing sends SMS |
| `google` | a signed identity token | yes, fully |
| `apple` | a signed identity token | yes, fully |

One person can have several. `identities` is a separate table keyed on
`(provider, subject)`, which matters most for Apple: **Apple sends an email
address on the first authorization only.** An account keyed on email would
therefore create a *second* account on that person's second sign-in and strand
everything they had. It is the single most common way this gets built wrong.

Two consequences worth stating plainly:

* **An account may have no email and no phone at all.** Sign in with Apple,
  decline to share the address, and that is a complete, valid account. Anything
  that treats "has an email" as "is signed in" is wrong - `Listener` carries an
  explicit `has_account` for exactly this.
* **An Apple address may be a private relay.** It forwards today and its owner
  can switch it off tomorrow, so nothing may promise to reach somebody there.
  The provider endpoint returns `private_relay` so the interface can say so.

### What is deliberately not verified, and say so out loud

Neither the email nor the phone number is verified, because the app has no
route to deliver a message to either. That means:

* **A phone number is an identifier, not a second factor.** Sign-up proves
  possession of a password and nothing else.
* **There is still no password reset**, and now it can be reached two ways
  instead of one. A forgotten password on an email-or-phone account is a lost
  account.

Both are honest gaps, both are the same missing capability - delivery - and the
sign-up screen has to say so rather than letting someone find out the hard way.
Sign in with Google or Apple has no such gap, which is a real argument for
making them the prominent options in the app.

### Verifying a provider token

The app gets an identity token from the platform SDK and posts it to
`POST /api/auth/provider`. The server checks the **signature** against the
provider's published keys, the **issuer**, the **audience** (our client ids),
the **expiry** and the **nonce**. There is no code exchange and no client
secret, because a native app needs neither - which removes the most common way
this is built wrong, a client secret shipped inside the app.

**A token is either verified or refused; there is no third branch.** Not
"decode it without checking if the library is missing", not "trust it in
development". A JWT is a plain base64 envelope: anyone can write one saying
they are anybody. So when `PyJWT` is absent, or its crypto backend is broken,
or no audience is configured, that provider is **unavailable** - `/api/health`
says which and why, and a sign-in attempt gets a 503 naming the fix. It never
degrades into accepting something it could not check.

That is the one place in this codebase where the usual "announce the fallback"
rule is not enough, because the quiet version hands over every account.

## Tiers

| | `free` | `plus` | `unlimited` |
|---|---|---|---|
| Episodes | 5 / day | 150 / week | no ceiling |
| Explore replays | 25 / day | no ceiling | no ceiling |
| Everything else | all of it | all of it | all of it |

The numbers are env-tunable (`FREE_EPISODES_PER_DAY` and friends) and they are
**placeholders in the honest sense**: numbers to start a free beta with, not
numbers anybody has measured. `python tools/usage_report.py` prints the median
and the p99 cost per listener, and on the one measured run the worst listener
cost 68x the median. The first real setting of these should come from that
report rather than from taste.

Three decisions inside that table:

* **Episodes and Explore replays are counted separately**, because they cost
  differently. An episode may write a script - a Claude call, ~$0.03 - and
  always synthesises audio. A replay provably *cannot* write one: the pipeline
  refuses. One combined allowance would price the cheapest surface as if it
  were the most expensive.
* **Free is capped per day and Plus per week.** A daily cap tells the listener
  enjoying it most to come back tomorrow. That is acceptable for a free tier
  and it is the wrong way to treat somebody who is paying for a Sunday
  afternoon of listening.
* **A cache hit still counts.** The listener heard an episode and the GPU
  produced it; only the model call was saved. An early version refunded
  whenever no Claude call had happened, which meant the limit worked on a cold
  server and quietly stopped working as the cache warmed - the worst possible
  way for a spending control to fail.

### Features, and why every tier has all of them today

`entitlements.FEATURES` is a registry of capabilities, each naming the lowest
tier that has it. **Today every feature is available on every tier.** That is
not an empty policy nobody got round to filling in; it is the only safe
starting state.

Every listener who has ever used FAM has had all of it. Shipping a tier system
that silently takes things away is the "quietly worse than intended" failure
this project has lost the most time to, with the twist that here the listener
notices and is right. So the *mechanism* ships on and the *policy* ships empty:
moving a feature behind a tier is one line, and
`tests/test_entitlements.py::test_every_feature_is_available_on_every_tier_today`
fails when somebody does it - not to prevent it, but to make it a decision
somebody made and wrote down.

Each feature carries a note arguing for or against gating it. The strongest
candidates, written while the reasoning is fresh: **live research** (Exa is
billed per call and is the one per-episode cost that is not the model) and
**API access** (it is the feature that lets somebody else's software spend the
GPU). The worst candidate is **Explore New** - gating discovery makes the free
tier a smaller world rather than a cheaper one.

### What is not here: payment

No receipts, no subscription lifecycle, no App Store server notifications, no
proration. `ACCOUNTS.set_plan` moves an account between tiers and that is all;
what decides *when* to call it does not exist.

That is deliberate sequencing rather than an omission. The enforcement path can
be finished, tested and trusted before any money moves, and when a payment
provider is chosen it plugs into one function instead of into every endpoint.
The ledger is already per-listener, so entitlement becomes a lookup rather than
a rebuild.

`/api/plans` deliberately carries **no prices**. A price here and a price in
App Store Connect are two places for one fact, the wrong one is always the one
the listener is reading, and only the store's own product metadata can be right
per country.

## Quotas: how the counting works

Calendar windows in UTC - a day is a UTC day, a week a UTC ISO week. Rolling
windows are fairer but cannot answer "when do I get more?" with a time somebody
can plan around, and a ceiling nobody can see the end of reads as a fault
rather than a limit. The cost of that choice, stated so it is a known trade: a
listener in California gets their reset in the late afternoon. Every verdict
carries `resets_at` so the interface can say when in their own clock.

A spend is **reserved first**, atomically, and refunded if the episode never
happened. Check-then-generate double-spends under concurrency: two requests see
the same count and both pass, and the second one has already cost a Claude
call by the time anybody notices.

A refusal is a **429** carrying the whole verdict in `X-FAM-Quota` - what the
limit was, what is left, when it resets, which tier. 402 would be the pedantic
status for "you have run out of allowance", but it means "payment required" in
a way nothing has ever agreed on, and 429 is what a client library already
knows to back off from.

### What quotas cannot do, and it matters

An anonymous listener is a session cookie, and a cookie can be thrown away.
Somebody who wants more than the free tier allows can clear it and start again.
The id is server-minted so it cannot be *forged*, but it can be *abandoned*.

That is a direct consequence of the settled constraint that an account gates
what is kept and never what is heard; the alternative is a login in front of
the first word. So the free quota is **a budget shaped like a limit, not a
security control**. If the beta shows it being routinely walked around, the
answers are device attestation or requiring an account to generate - both
product decisions, neither of which belongs in a counter.

## Deleting an account

`DELETE /api/account`, and it is a real feature rather than paperwork: App
Store guideline 5.1.1(v) requires any app offering account creation to offer
in-app deletion. `erase_listener` walks every store, and each store owns its
own `forget()` - a central deleter reaching into six databases by table name is
one that silently stops covering the seventh.

| | what happens |
|---|---|
| events, mixes, echoes, profile, preferences, attachments, quota counters | deleted |
| credentials, identities, sessions | deleted |
| **cost ledger** | **anonymised, not deleted** |
| **shared script cache** | **untouched, and needs no decision** |

The last two are the interesting ones.

**The ledger is anonymised.** What the GPU and the model cost in June is a fact
about the business, not personal data about a listener, and a ledger with holes
cannot be reconciled against a provider invoice - which is the entire job it
was built for. So the link to the person goes and the amount stays, joining the
existing bucket that already holds every session-less request. The rule that
falls out of this: nothing that could re-identify those rows may be added to
that table.

**The cache needs no decision at all.** It holds no `user_id` - it never has -
so a script written for this listener is already unattributed. Other listeners'
Explore feeds do not develop holes because somebody left. (An earlier note in
`IOS_APP.md` framed this as a decision to be made; it turned out to be already
made by the schema, which is the better outcome.)

Deletion does not ask for a password, on purpose: an account created with Sign
in with Apple has none, and a confirmation step only some accounts can satisfy
is a deletion route only some accounts have. Holding the session is the proof.
Afterwards the session is dropped, the next request mints a fresh anonymous
listener, and the app keeps working.

## The public API

Everything is reachable at `/api/v1/...` as well as at `/api/...`. The prefix
is a **rewrite**, not a second set of routes, because two registrations of one
endpoint are two places for a decorator to drift - and the drift would show up
as a native client quietly getting different behaviour from the web one.

Why a version at all: an app on somebody's phone cannot be redeployed with the
server. The day an endpoint has to change shape, `/api/v2` can carry the new
one while `/api/v1` keeps the promise made to every copy already installed.
Without a prefix that day forces a choice between breaking installed apps and
never changing the API.

### Two carriers, one session

The web client uses the HttpOnly cookie and cannot read it, which is what makes
an XSS unable to walk off with somebody's identity. A native client asks for
`want_token: true` on sign-up or login, stores the returned token in the
Keychain, and sends it as `Authorization: Bearer`.

**The settled rule is untouched: the id still never comes from the client.** A
bearer token is the same server-minted, high-entropy, revocable,
never-stored-in-the-clear string the cookie carries. Only the envelope changes.

A browser must never ask for the token - reading it in script is precisely what
HttpOnly exists to prevent - which is why it is an explicit opt-in rather than
something every response carries. When both are sent the cookie wins.

### CORS

Off unless `API_ORIGINS` names an origin, and never `*`: these requests carry
credentials, and a browser refuses a wildcard together with credentials - so a
wildcard would look permissive, not work, and hide the real fix. A native app
is not a browser, sends no `Origin`, and needs nothing here.

### The endpoints this added

    POST   /api/auth/signup           email or phone, + want_token
    POST   /api/auth/login            email or phone, + want_token
    POST   /api/auth/provider         a verified Google or Apple identity token
    POST   /api/auth/password/set     a first password, for a provider account
    GET    /api/account               settings: identities, sessions, tier, usage
    POST   /api/account               change name, email, phone
    DELETE /api/account               erase everything (5.1.1(v))
    POST   /api/account/signout-everywhere
    DELETE /api/account/identity?provider=…
    GET    /api/entitlements          this listener's tier, limits and what is left
    GET    /api/plans                 every tier and feature, for a pricing screen

`/api/health` gained `api`, `oauth` and `quotas` blocks: the version prefix,
whether each provider can actually complete a sign-in and why not, and whether
limits are being enforced. All three follow **verify, do not inspect**
(PROBLEMS.md §52) - a server with limits switched off looks identical from the
outside to one with them on, and that is exactly the thing worth being able to
ask.

## What to do next

1. **Set the limits from the report, not from taste.** `usage_report.py`
   already prints what a listener costs. Until that happens the numbers above
   are guesses that happen to be running.
2. **Decide which features the free tier does not get** - and change exactly
   one line per decision, plus the test that guards it.
3. **Payment**, when there is something to sell. StoreKit 2 on the app side,
   App Store server notifications into `set_plan` on this side.
4. **Delivery** - email or SMS - which is the single missing capability behind
   password reset, address verification and phone verification all at once.
