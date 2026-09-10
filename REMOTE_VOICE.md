# The voice on a rented card, and the app where it is cheap

Chatterbox needs a GPU. The rest of FAM — Claude, the cache, eight SQLite
stores, the interface — needs a web server. Running both on the same machine
means paying GPU prices for the 99% of the time nothing is being synthesised,
which `DEPLOY.md` names as the constraint to plan around:

> **Chatterbox runs in-process**, so every replica needs a GPU, and a GPU left
> running is the expensive kind.

This splits them. The app runs on Render; the voice runs on RunPod; between
them is one JSON contract. **The audio is unchanged** — same weights, same
`reference_3.wav`, same six generation settings — because the worker imports
the real `ChatterboxEngine` rather than reimplementing it. What changes is
where the card is and what it costs.

    Render (CPU, always on)              RunPod (GPU)
    ┌───────────────────────────┐        ┌──────────────────────────┐
    │ app.py                    │        │ voice_worker/            │
    │  Claude, cache, DBs,      │ ─────▶ │  synth.py                │
    │  player, /api/audio       │  HTTP  │   └ ChatterboxEngine     │
    │ remote_voice.py           │ ◀───── │  network volume:         │
    │  RemoteChatterboxEngine   │  PCM   │   weights + reference_3  │
    └───────────────────────────┘        └──────────────────────────┘

## Two ways to rent the same card

One worker image serves both. Moving between them is **two environment
variables on the app** — never a rebuild, never a code change.

| | `REMOTE_VOICE_TRANSPORT=runpod` | `REMOTE_VOICE_TRANSPORT=http` |
|---|---|---|
| What it is | RunPod Serverless | An always-on RunPod pod |
| Billing | per second of execution | per hour, speaking or not |
| Cold start | yes, when scaled to zero | never |
| Cheaper below | ~4 hours of audio a day | above that |
| Also set | `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY` | `REMOTE_VOICE_URL`, `REMOTE_VOICE_TOKEN` |

And a third position that needs no worker at all: **`VOICE_BACKEND=chatterbox`**,
which is the default and is the in-process card `Dockerfile.gpu` deploys.
Nothing in this document removes that path — it is the destination, and this is
the detour taken while the volume does not justify a full-time GPU.

## Once, ever

### 1. Build and push the worker image

    docker build -f Dockerfile.voice -t <registry>/fam-voice:1 .
    docker push <registry>/fam-voice:1

### 2. Create a RunPod network volume, and put the voice on it

In the RunPod console: **Storage → Network Volume**, 20 GB is plenty, in the
region you will run workers in. Then attach it to any cheap pod once and copy
two files into it:

    scp reference_3.wav         root@<pod>:/state/voices/
    scp reference_3.rights.json root@<pod>:/state/voices/

The rights record must clear three fields, each `"yes"`:

    {"consent": "yes", "commercial_use": "yes", "synthetic_voice_cleared": "yes"}

The engine refuses to speak without both files, which is why a worker that
"has the image" still cannot talk. A cloned voice is somebody's voice, and that
stays true on rented hardware — which is also why neither file is in the image.

The volume mounts at `/state`, so the same tree does double duty: `/state/hf`
takes the Chatterbox weights on first use. **That download is most of a cold
start**, and paying it once on a volume rather than once per worker is what
makes serverless viable at all.

### 3. Create the endpoint

**Serverless → New Endpoint**, pointing at `<registry>/fam-voice:1`.

| Setting | Value | Why |
|---|---|---|
| GPU | 24 GB class (A5000, 4090, L4) | what Chatterbox was measured on |
| Network volume | the one from step 2 | weights and the voice |
| Active workers | 0 | the whole point; nothing idle is billed |
| Max workers | 1–2 | one card serves one generation at a time |
| Idle timeout | 60s | long enough that a second episode is warm |
| FlashBoot | on | it is what makes a warm start ~instant |
| Env | `VOICE_WORKER_MODE=serverless` | the image's default; set it anyway |

Copy the **endpoint id**.

*For an always-on pod instead:* deploy the same image as a Pod, set
`VOICE_WORKER_MODE=http` and `REMOTE_VOICE_TOKEN=<a long random string>`,
expose port 8001, and use the proxy URL RunPod gives you as
`REMOTE_VOICE_URL`.

### 4. Point Render at it

`render.yaml` already declares these; set the two `sync: false` values in the
Render dashboard (**Environment**):

    VOICE_BACKEND=remote
    REMOTE_VOICE_TRANSPORT=runpod
    RUNPOD_ENDPOINT_ID=<from step 3>
    RUNPOD_API_KEY=<a RunPod API key>
    REMOTE_VOICE_SAMPLE_RATE=24000

`RUNPOD_API_KEY` goes through the same credential chain as everything else, so
`FAM_SECRETS` works instead and is better: rotating the key stops being an edit
to a dashboard. See `CREDENTIALS.md`.

Render redeploys on push, so there is nothing else to do.

## Before you trust it

    python tools/demo_preflight.py     # names which of the four are missing
    python verify_voice.py             # actually synthesises

`verify_voice.py` is the one that matters, because it performs the real action
rather than confirming a variable is set — the distinction PROBLEMS.md §52 is
about. From the Render service, `GET /api/health` reports the same thing:

    "tts": {
      "backend": "remote",
      "selected": "remote",
      "interim": false,
      "remote": {
        "configured": true,
        "transport": "runpod",
        "endpoint": "https://api.runpod.ai/v2/<id>",
        "reachable": {"state": "ok", "latency_seconds": 2.1}
      }
    }

**`configured` and `reachable` are different questions and are reported
separately.** `configured: true, reachable: unknown` means nothing has ever
actually spoken; `reachable: failed` carries the reason. A health check that
only read configuration would answer the cheaper question and say OK.

## The cold start, and what is done about it

A serverless worker at zero pays container boot plus a ~10s model load before
its first word. That is exactly the wait the one-sentence spec refuses.

It is answered the way CLAUDE.md says to answer latency — **by starting
earlier, not by filling the gap**. When a request arrives, `app.py` fires
`remote_voice.wake()`: a throwaway job that boots a worker and loads the model,
sent *before Claude has written a word*. The script takes several seconds to
write, and the worker boots during them.

The wake is a hint, not a mechanism. It never raises, never blocks, and never
delays a request; a miss costs only the cold start it was trying to hide. It
also will not stampede — a worker that is already booting does not boot faster
for being asked twice.

If measurement says the wake is not covering it, in order of preference:

1. **Raise the idle timeout.** The cheapest fix by far: it only bills while a
   worker is up, and a listener who plays two episodes gets the second warm.
2. **One active worker during peak hours.** This is a pod wearing a different
   hat — it is billed continuously — so price it as one.
3. **Switch to `REMOTE_VOICE_TRANSPORT=http`.** At that point you are paying
   for an always-on card and should compare against `Dockerfile.gpu` directly.

## What this does *not* do

**No fallback, ever.** A remote voice that fails raises with the reason
attached — it never becomes a local engine, a different voice, or silence. That
is PROBLEMS.md §61's second guard, re-added by hand as it said to. When nothing
can speak the honest states are still exactly two: Chatterbox speaks, or a
placeholder tone plays and everything says so.

**It is never the default.** `VOICE_BACKEND` defaults to `chatterbox` and
nothing auto-detects. Merely having a RunPod key in the environment does not
make a rented GPU what every listener gets — §61's first guard, which is how
WellSaid silently became the default voice on every machine without Piper.

**Nothing here starts, stops, resizes or pays for a pod.** `RUNPOD_PRODUCTION.md`
said that and it stays true. This makes the machine reproducible; it does not
make it automatic.

**Bandwidth is unchanged and still the thing that bites at scale**: 2.65 MB/min
per listener at 22050 Hz, more at Chatterbox's 24000. Opus over the stream is
the fix and is compatible with the no-audio-files rule — compression is fine,
writing a *file* is not.

## Going back to a single GPU box

Unset `VOICE_BACKEND` (or set it to `chatterbox`) and deploy `Dockerfile.gpu`
as `DEPLOY.md` describes. That is the whole procedure. Nothing about the
in-process path was removed, deprecated or altered to make room for this:
`Dockerfile.gpu`, `RUNPOD_PRODUCTION.md`, `tools/pack_for_pod.py`,
`tools/pod_production_test.sh`, `requirements-chatterbox.txt` and
`ChatterboxEngine` itself are all untouched, and
`tests/test_remote_voice.py::test_the_remote_voice_is_not_reachable_without_being_asked_for`
fails if the default ever drifts away from them.

This is a deliberate exception to the project's usual habit of deleting a thing
rather than switching it off. That rule exists for things that should not come
back — the cold open, Piper, WellSaid. A card of your own is where this is
going, so here the knob **is** the point.
