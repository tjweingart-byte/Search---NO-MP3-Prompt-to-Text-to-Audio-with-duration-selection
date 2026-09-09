# Running FAM as a service

This replaces the pod ritual — `pack_for_pod.py`, scp a tarball, extract, pip
install, export keys, run the harness. That loop is right for renting a card
for an hour to *measure* something. It is wrong for a service: every step is a
chance to set it up differently from last time, and none of it is recorded.

The split that makes it repeatable:

| baked into the image | set once on the host | never anywhere |
|---|---|---|
| CUDA, torch, chatterbox, exa, the app | `FAM_SECRETS` (or the two keys) | credentials in the image |
| the validated settings (phase6, exa, ANSWER_FIRST=1) | the persistent volume | the voice in the repo |

**Prefer `FAM_SECRETS` to the two keys.** It is one *non-secret* variable that
names where the credentials live, so the host template carries no secret at all,
a new pod fetches its own, and rotating a key is a change in the manager rather
than an edit to every template. `CREDENTIALS.md` has the recipes.

## The one setting the image pins against the code default

`ANSWER_FIRST=1`, and it is deliberate.

`config.py` derives that value from the research backend — Claude's own search
is slow enough to need the from-knowledge cover, Exa is not, so `exa` implies
`answer_first=False`. That reasoning stands, and the image does not change it.

But the configuration that was **listened to and judged good** — Phase 6,
Chatterbox, `reference_3`, ~4.5s and ~2.992s to first audio, research handing
off mid-episode — ran with the cover **on**. It predates that derivation
(commit `91d9dad`), and `tools/pod_production_test.sh` never set the variable,
so it inherited a default that has since flipped. An image without this line
would deploy a configuration nobody has heard.

So: the image reproduces what was validated; the code default keeps its own
reasoning for every other deployment. A container from this image resolves

    answer_first True · answer_first_share 0.5 · phase6 · exa

which is the four settings the good run had.

**This line is provisional.** Run

    ANSWER_FIRST=1 bash tools/pod_production_test.sh

on a card, compare against the same harness without it, and let the numbers
decide whether the cover belongs in `config.py` — at which point this pin
becomes redundant and should go.

## Once, ever

**1. Build and push the image.**

    docker build -f Dockerfile.gpu -t <registry>/fam:<tag> .
    docker push <registry>/fam:<tag>

**2. Create a persistent volume** and mount it at `/state`. On RunPod this is a
network volume attached to the template; on any other host it is a normal
volume. One mount, three trees:

    /state/hf       Chatterbox weights
    /state/voices   reference_3.wav + reference_3.rights.json
    /state/data     the eight SQLite stores

**3. Put the voice on the volume, once.** The engine refuses to speak without
both files — a cloned voice is somebody's voice:

    scp reference_3.wav        <host>:/state/voices/
    scp reference_3.rights.json <host>:/state/voices/

The rights record must clear three fields, each `"yes"`:

    {"consent": "yes", "commercial_use": "yes", "synthetic_voice_cleared": "yes"}

**4. Set the credentials on the template**, as environment variables. Not typed
into a shell on the pod - that is the step you are trying to stop repeating.

One line is enough, and it is not a secret:

    FAM_SECRETS=cmd:aws secretsmanager get-secret-value --secret-id fam --query SecretString --output text

The pod authenticates as itself - an IAM role, a service account - and fetches
`ANTHROPIC_API_KEY` and `EXA_API_KEY` at startup. Setting the two keys directly
still works and is one step simpler; what it costs is that rotating either one
means editing the template and restarting, where the provider is picked up by a
running server the next time a key is rejected. See `CREDENTIALS.md`.

## Every deploy after that

Start the container. That is the whole procedure. Nothing is installed,
nothing is uploaded, nothing is typed.

The first pod to run pays a one-time weight download into `/state/hf`; every
pod after that starts with them already there. That download is the
`up after 38s (warm-up included)` line in the pod logs — it is infrastructure
boot, not request latency, and the volume is what stops you paying it again.

## Before you trust it, one command

    python tools/demo_preflight.py

It reports all four ways this can look like it is working when it is not —
writing, speech, research, cache — and names which is missing. A deployment
that passes it will speak; one that does not will say why in a sentence.

`python verify_voice.py` goes further and actually synthesises, which is the
difference between "Chatterbox is installed" and "this machine can speak".

## What is deliberately still manual

**The voice and its rights record.** They are per-machine state and stay out
of the image on purpose: a cloned voice in a container registry is somebody's
voice in a container registry. Once on the volume, they persist.

**Starting and paying for the machine.** `RUNPOD_PRODUCTION.md` says nothing in
this repository starts, stops, resizes or pays for a pod, and that stays true.
This makes the machine reproducible; it does not make it automatic.

## Scaling past one box

The constraint to know before you plan around it: **Chatterbox runs in-process**,
so every replica needs a GPU, and a GPU left running is the expensive kind.
Splitting speech into its own service — prototyped as
`experiments/adapters/chatterbox_server_example.py` on the unmerged
`fam-repo-inventory` branch — is what lets the app run on cheap CPU hosting with
only the voice on a card. That is a real decision, not a config change.

The other number that bites at scale is bandwidth: **2.65 MB/min uncompressed**
per listener. Opus over the stream is the fix and is compatible with the
no-audio-files constraint — compression is fine, writing a *file* is not.
