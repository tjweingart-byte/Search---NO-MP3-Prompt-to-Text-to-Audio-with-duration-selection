# Running FAM as a service

This replaces the pod ritual — `pack_for_pod.py`, scp a tarball, extract, pip
install, export keys, run the harness. That loop is right for renting a card
for an hour to *measure* something. It is wrong for a service: every step is a
chance to set it up differently from last time, and none of it is recorded.

The split that makes it repeatable:

| baked into the image | set once on the host | never anywhere |
|---|---|---|
| CUDA, torch, chatterbox, exa, the app | `ANTHROPIC_API_KEY`, `EXA_API_KEY` | credentials in the image |
| the validated defaults (phase6, exa) | the persistent volume | the voice in the repo |

## Once, ever

**1. Build and push the image.**

    docker build -f Dockerfile.gpu -t <registry>/fam:<tag> .
    docker push <registry>/fam:<tag>

**2. Create a persistent volume** and mount it at `/state`. On RunPod this is a
network volume attached to the template; on any other host it is a normal
volume. One mount, three trees:

    /state/hf       Chatterbox weights
    /state/voices   reference_3.wav + reference_3.rights.json
    /state/data     the six SQLite stores

**3. Put the voice on the volume, once.** The engine refuses to speak without
both files — a cloned voice is somebody's voice:

    scp reference_3.wav        <host>:/state/voices/
    scp reference_3.rights.json <host>:/state/voices/

The rights record must clear three fields, each `"yes"`:

    {"consent": "yes", "commercial_use": "yes", "synthetic_voice_cleared": "yes"}

**4. Set the two keys on the template**, as environment variables. Not typed
into a shell on the pod — that is the step you are trying to stop repeating.

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
