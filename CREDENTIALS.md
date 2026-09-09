# Credentials, and never typing one again

`~/.fam/env` (PROBLEMS.md §53) stopped the key being re-pasted **on one
machine**. The demo does not run on one machine. It runs on a rented GPU pod,
a fresh container, a colleague's laptop, a CI runner — and every one of those
is a machine with no `~/.fam/env` in it, so the key gets typed again and §53's
finding repeats one layer up.

The fix is §53's, moved one level out. Put the credential where the *deployment*
can reach it programmatically, and let the app fetch it when it needs it.

```
FAM_SECRETS='cmd:aws secretsmanager get-secret-value --secret-id fam --query SecretString --output text'
```

That is **one non-secret variable**, set once on the host template or baked
into an image. It is the only line a machine that has never run FAM needs, and
nothing is typed on that machine ever again.

---

## The chain

Highest priority first. Everything below the first line is a way of not having
to do the first line.

| | source | set by | good for |
|---|---|---|---|
| 1 | the process environment | a platform dashboard, a CI secret, `docker -e` | one deployment |
| 2 | **`FAM_SECRETS`** | a secrets manager, read at runtime | **every machine, and rotation** |
| 3 | the project `.env` | a person, once per checkout | pinning one project's key |
| 4 | `~/.fam/env` | `python setup_key.py` | one laptop (§53) |

An explicit environment variable always wins, because somebody who exported one
meant it — usually to test one specific key against one specific bug. A refresh
will not overwrite it either.

`FAM_SECRETS` may itself be named in a `.env`, which is how a laptop configures
this once. It is not a secret: it says *where* the secrets are.

---

## The provider is a command, deliberately

There is no AWS SDK here, no Vault client, no Doppler library — one hook that
shells out to whatever the deployment already has. Two schemes:

```
file:<path>                     a Docker or Kubernetes secret file
file:NAME=<path>                one file that holds one bare value
cmd:<command>                   anything that prints secrets
cmd:NAME=<command>              ... and prints one bare value
```

Output is parsed as a **JSON object** or as **dotenv lines**, which between them
is what every manager below emits. A `NAME=` prefix binds the whole stripped
output to one variable, for the managers that return a bare secret.

### Recipes

```sh
# AWS Secrets Manager — authenticates as the instance's IAM role
FAM_SECRETS='cmd:aws secretsmanager get-secret-value --secret-id fam --query SecretString --output text'

# Google Secret Manager — as the workload's service account
FAM_SECRETS='cmd:gcloud secrets versions access latest --secret=fam'

# HashiCorp Vault
FAM_SECRETS='cmd:vault kv get -format=json -field=data secret/fam'

# Doppler
FAM_SECRETS='cmd:doppler secrets download --no-file --format json'

# 1Password — one secret per line
FAM_SECRETS='cmd:ANTHROPIC_API_KEY=op read op://vault/fam/anthropic
cmd:EXA_API_KEY=op read op://vault/fam/exa'

# Docker / Kubernetes secrets, mounted as files
FAM_SECRETS='file:ANTHROPIC_API_KEY=/run/secrets/anthropic'
```

Every one of those authenticates as **the machine** — an IAM role, a service
account, a workload identity — not as a stored password. Nothing here is a
credential that itself has to be delivered somehow.

One entry per line; a later line overrides an earlier one, the way sourcing two
files does. A command that genuinely needs a leading assignment writes
`cmd:env FOO=bar ./thing`, because `NAME=` in front is read as a binding.

**No new dependency, and no vendor chosen on the app's behalf.** The escape
hatch for a manager nobody has thought of yet is that it is already a shell
command.

---

## The demo, which is what this was for

On a machine that has never run FAM:

```sh
export FAM_SECRETS='cmd:doppler secrets download --no-file --format json'
./demo.sh
```

`demo.sh` no longer tests `$ANTHROPIC_API_KEY` — that is only what *the shell*
can see, and three of the four sources above are resolved in Python. It asks
the app whether a key will be found, which is the actual question. With a
provider set it never prompts; with nothing attached to the terminal it prints
the `FAM_SECRETS` line instead of asking a pod to type something.

**CI already needs no key** and this changes nothing there. The suite is
hermetic by construction (`tests/conftest.py` sets `FAM_IGNORE_DOTENV`, and the
provider is skipped under it too), so `./dev.sh check` and
`.github/workflows/ci.yml` run green with no credential at all. If you were
about to add `ANTHROPIC_API_KEY` to the repository's GitHub secrets: don't. It
would buy nothing and add a key to rotate.

---

## Rotation, without a redeploy

`credentials.refresh()` re-runs the provider and replaces the values it owns, in
the running process. It happens at startup, on `FAM_SECRETS_TTL` if you set one,
and — the useful one — **when Claude rejects the key in force**, which is the
moment the app finds out a rotation happened.

So rotating is: change it in the manager. No code change, no redeploy, no
restart, no downtime. `tests/test_credentials.py` pins that a key rotated away
under a running server is picked up on the next verification rather than
reported as a bad key.

This only works if the value comes from the provider. A key baked into the
container's environment is a snapshot: rotating the manager does not reach it.
That is the trade for using source 1 instead of source 2, and it is fine for a
deployment that redeploys often.

---

## Pools, and what they actually buy

Several keys, comma separated:

```sh
ANTHROPIC_API_KEYS=sk-ant-a,sk-ant-b
EXA_API_KEYS=exa-a,exa-b
```

A key that is rejected steps aside and the next one takes over, published into
the environment so the Anthropic SDK and `research.py` both follow it without
either of them knowing this module exists.

**Be precise about what that buys, because the obvious assumption is wrong:**

* **Anthropic rate limits are per organisation, not per key.** A second key from
  the same org shares one bucket and buys **no extra headroom at all**. What a
  pool buys here is *failover*: a key that is revoked, expired or spend-capped
  halfway through a demo steps aside instead of ending the demo. Real headroom
  is a higher usage tier, or separate workspaces — not more keys.
* **Exa's limits are per key** (10 rps on `/search` and `/answer` on standard
  plans). There a pool is genuine headroom.

One key is a pool of one and behaves exactly as it did before any of this
existed.

---

## What this does *not* solve

Worth saying plainly, because "we fixed credentials" is easy to hear as "we
fixed scale", and the credential was never the binding constraint here.

* **GPU concurrency is the real ceiling.** Chatterbox runs in-process, so every
  replica needs a card, and one card serves a bounded number of simultaneous
  generations before they queue. No number of keys changes that.
  `DEPLOY.md` §"Scaling past one box" is the live version of this problem.
* **Bandwidth: 2.65 MB/min per listener, uncompressed.** Opus over the stream is
  the fix and is compatible with the no-audio-files constraint.
* **Per-listener metering does not exist.** With one key behind everyone, the
  provider only ever sees this account — it cannot tell one listener's spend
  from another's. Per-user limits, billing and abuse detection all need a row
  written in *our* database, tagged with the id from `_listener(request)`, at
  the moment each model call is made. Nothing here does that yet.
* **Listeners never supply their own key** and there is no plan for them to.
  Every call is made by the backend on their behalf; the key never reaches a
  browser. That is the settled architecture, not an interim state.

---

## Checking it, rather than assuming it

Per PROBLEMS.md §52: everything here reports the credential it actually
resolved, never that one was configured.

```sh
python setup_key.py --show        # source, fingerprint, provider state, and
                                  # whether Claude still accepts it
python tools/demo_preflight.py    # what this machine will do if you start now
curl -s localhost:8000/api/health | python -m json.tool
```

`/api/health` carries `credentials.secrets`: the provider, its last outcome,
which names it supplied, the pool sizes and the source of each credential. It
never carries a key — a report that prints a secret is a report nobody can
paste into a bug, which makes it a report nobody runs.

A provider that is set and broken is said out loud at startup, on `/api/health`
and in the preflight, and it never quietly falls through to the canned script.
A configured-and-failing provider is reported even when a key was found some
other way, because it means **the next machine will find nothing**.
