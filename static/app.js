/*
 * Streaming player.
 *
 * The server sends raw 16-bit PCM over a chunked response. We read it with the
 * fetch streams API, convert each chunk to float samples, and schedule it on a
 * Web Audio clock. Playback starts on the first chunk, so the listener hears
 * the opening line while the rest of the episode is still being written.
 *
 * No Blob, no object URL, no encoded file - the samples go straight to the
 * output device.
 */
const $ = (id) => document.getElementById(id);

const form = $("form");
const queryEl = $("query");
const minutesEl = $("minutes");
const minutesOut = $("minutesOut");
const goBtn = $("go");
const stopBtn = $("stop");
const statusEl = $("status");
const meterWrap = $("meterWrap");
const barEl = $("bar");
const elapsedEl = $("elapsed");
const totalEl = $("total");
const transcriptWrap = $("transcriptWrap");
const engineEl = $("engine");

let ctx = null;          // AudioContext
let controller = null;   // aborts the in-flight fetch
let playHead = 0;        // next scheduled start time, in AudioContext seconds
let startedAt = 0;
let ticker = null;

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function setMinutes() {
  const m = Number(minutesEl.value);
  minutesOut.textContent = `${m} minute${m === 1 ? "" : "s"}`;
  totalEl.textContent = fmt(m * 60);
}
minutesEl.addEventListener("input", setMinutes);
setMinutes();

function say(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

/* Several problems can be true at once - no API key AND no speech engine.
   Showing them one at a time means the later one silently overwrites the
   earlier one, and the user is told the wrong reason for what they see. */
function showNotices(notices) {
  statusEl.classList.remove("error");
  statusEl.innerHTML = "";
  for (const note of notices) {
    const p = document.createElement("p");
    p.className = note.blocking ? "blocking" : "warn";
    p.textContent = note.text;
    statusEl.appendChild(p);
  }
}

function reset() {
  if (ticker) { clearInterval(ticker); ticker = null; }
  if (controller) { controller.abort(); controller = null; }
  if (ctx) { ctx.close().catch(() => {}); ctx = null; }
  goBtn.disabled = false;
  goBtn.classList.remove("working");
  stopBtn.hidden = true;
}

stopBtn.addEventListener("click", () => { reset(); say("Stopped."); });

/* One PCM chunk -> one scheduled AudioBuffer, appended to the play head. */
function schedule(int16, sampleRate) {
  const buf = ctx.createBuffer(1, int16.length, sampleRate);
  const out = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 32768;

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);

  // Never schedule in the past: if the network stalled and the play head fell
  // behind the clock, restart from now instead of dropping the chunk.
  const when = Math.max(playHead, ctx.currentTime + 0.05);
  src.start(when);
  playHead = when + buf.duration;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  reset();

  const q = queryEl.value.trim();
  const minutes = Number(minutesEl.value);
  if (!q) return;

  goBtn.disabled = true;
  goBtn.classList.add("working");
  stopBtn.hidden = false;
  meterWrap.hidden = false;
  transcriptWrap.hidden = true;
  barEl.style.width = "0%";
  say("Researching and writing…");

  // Must be created inside the click handler or browsers keep it suspended.
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.resume();
  playHead = 0;
  controller = new AbortController();

  const url = `/api/audio?q=${encodeURIComponent(q)}&minutes=${minutes}&fmt=pcm`;

  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: `Request failed (${res.status})` }));
      throw new Error(body.error || `Request failed (${res.status})`);
    }

    const sampleRate = Number(res.headers.get("X-Sample-Rate")) || 22050;
    const target = Number(res.headers.get("X-Requested-Seconds")) || minutes * 60;
    totalEl.textContent = fmt(target);

    const reader = res.body.getReader();
    let leftover = new Uint8Array(0);
    let first = true;
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!controller) break;

      // 16-bit samples can straddle a chunk boundary; carry the odd byte over.
      let bytes = value;
      if (leftover.length) {
        const merged = new Uint8Array(leftover.length + value.length);
        merged.set(leftover, 0);
        merged.set(value, leftover.length);
        bytes = merged;
      }
      received += value.length;
      const usable = bytes.length - (bytes.length % 2);
      leftover = bytes.slice(usable);
      if (!usable) continue;

      schedule(new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + usable)),
               sampleRate);

      if (first) {
        first = false;
        startedAt = performance.now();
        say("Playing — the rest is still being written.");
        ticker = setInterval(() => {
          const elapsed = (performance.now() - startedAt) / 1000;
          elapsedEl.textContent = fmt(Math.min(elapsed, target));
          barEl.style.width = `${Math.min(100, (elapsed / target) * 100)}%`;
        }, 250);
      }
    }

    // A stream that ends with no samples is a server-side failure that could
    // not change the status code. Never report that as success.
    if (received === 0) {
      throw new Error("The server sent an empty episode. Check the server log — " +
                      "usually a missing ANTHROPIC_API_KEY or no speech engine installed.");
    }
    const expected = target * sampleRate * 2;
    if (received < expected * 0.5) {
      say("The episode ended early — generation failed partway through. Check the server log.", true);
    } else {
      say("Episode complete.");
    }
    // Let the tail finish playing before tearing the context down.
    const remaining = Math.max(0, playHead - ctx.currentTime) * 1000;
    setTimeout(() => {
      if (ticker) clearInterval(ticker);
      goBtn.disabled = false;
      goBtn.classList.remove("working");
      stopBtn.hidden = true;
    }, remaining + 250);
  } catch (err) {
    if (err.name === "AbortError") return;
    say(err.message || "Something went wrong.", true);
    reset();
  }
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    engineEl.textContent = `${h.model} · voice: ${h.tts.selected} · ${h.sample_rate} Hz`;

    const notices = [];
    if (!h.api_key_configured) {
      notices.push({
        blocking: true,
        text: "Listen is disabled: the server has no Anthropic credentials. Put " +
              "ANTHROPIC_API_KEY in a .env file in the folder you run ./run.sh from, " +
              "then restart the server.",
      });
    }
    if (h.tts.selected === "debug") {
      notices.push({
        blocking: false,
        text: "No speech engine installed, so you would hear a placeholder tone " +
              "rather than a voice. Install espeak-ng (sudo apt-get install espeak-ng, " +
              "or brew install espeak-ng) and restart the server.",
      });
    }

    const blocking = notices.find((n) => n.blocking);
    goBtn.disabled = Boolean(blocking);
    // Hovering a disabled control should explain itself.
    goBtn.title = blocking ? blocking.text : "";
    if (notices.length) showNotices(notices);
  })
  .catch(() => say("Could not reach the server. Is it still running?", true));
