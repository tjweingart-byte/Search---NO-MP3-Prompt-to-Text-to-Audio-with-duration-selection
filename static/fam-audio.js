/*
 * FamAudio - streaming speech with transport controls.
 *
 * The first version scheduled each incoming chunk and forgot it, which is the
 * cheapest way to play a live stream but makes seeking impossible: there is
 * nothing behind you to go back to. This version keeps every sample it has
 * received in a growing buffer and drives playback from a position cursor, so
 * skip, scrub and speed changes all work on audio that is still arriving.
 *
 * Samples are kept as Int16 (about 2.6 MB per minute) and converted to float
 * only for the short slice being scheduled, which halves the memory a long
 * episode would otherwise hold.
 *
 * Public surface:
 *   FamAudio.play(query, minutes, handlers)
 *   FamAudio.pause() / resume() / stop() / isPaused()
 *   FamAudio.skip(seconds)          - relative, negative to go back
 *   FamAudio.seek(seconds)          - absolute
 *   FamAudio.setRate(multiplier)    - 1 = normal, 1.5 = half again as fast
 *   FamAudio.position() / duration() / isActive()
 */
window.FamAudio = (function () {
  "use strict";

  // How far ahead of the clock to keep audio scheduled, and how much to
  // schedule at once. Short slices keep seek and rate changes responsive.
  var LOOKAHEAD = 0.35;
  var SLICE = 0.25;
  // While the episode is still being written, never seek closer than this to
  // the end of what has arrived. Landing exactly on the edge starves the
  // player: nothing is left to schedule, playback stops dead, and further
  // skips appear to do nothing because the cursor is already pinned there.
  var TAIL_MARGIN = 2.0;

  var ctx = null;
  var controller = null;
  var timer = null;
  var handlers = {};
  var token = 0;

  var pcm = null;          // Int16Array, grown as the stream arrives
  var totalSamples = 0;    // how much of `pcm` is real audio
  var sampleRate = 22050;
  var cursor = 0;          // next sample to schedule
  var playHead = 0;        // ctx time at which `cursor` will be heard
  var rate = 1;
  var sources = [];
  var streamDone = false;
  var ended = false;
  var active = false;

  function reset() {
    if (timer) { clearInterval(timer); timer = null; }
    if (controller) { try { controller.abort(); } catch (e) {} controller = null; }
    stopSources();
    if (ctx) { var c = ctx; ctx = null; c.close().catch(function () {}); }
    pcm = null; totalSamples = 0; cursor = 0; playHead = 0;
    streamDone = false; ended = false; active = false; rate = 1;
  }

  function stopSources() {
    for (var i = 0; i < sources.length; i++) {
      try { sources[i].onended = null; sources[i].stop(); } catch (e) {}
    }
    sources = [];
  }

  function append(int16) {
    if (!pcm) pcm = new Int16Array(sampleRate * 60);
    if (totalSamples + int16.length > pcm.length) {
      var bigger = new Int16Array(Math.max(pcm.length * 2, totalSamples + int16.length));
      bigger.set(pcm.subarray(0, totalSamples), 0);
      pcm = bigger;
    }
    pcm.set(int16, totalSamples);
    totalSamples += int16.length;
  }

  /* Current playback position in samples, derived from the audio clock so it
     stays correct across pauses (a suspended context stops advancing). */
  function positionSamples() {
    if (!ctx) return 0;
    var aheadSeconds = Math.max(0, playHead - ctx.currentTime);
    return Math.max(0, cursor - aheadSeconds * rate * sampleRate);
  }

  /* Keep the clock fed. Runs on a timer so it also picks up newly arrived
     audio after the buffer has run dry. */
  function tick() {
    if (!ctx || !active) return;

    while (playHead - ctx.currentTime < LOOKAHEAD && cursor < totalSamples) {
      var end = Math.min(cursor + Math.floor(SLICE * sampleRate), totalSamples);
      var length = end - cursor;
      if (length <= 0) break;

      var buf = ctx.createBuffer(1, length, sampleRate);
      var out = buf.getChannelData(0);
      for (var i = 0; i < length; i++) out[i] = pcm[cursor + i] / 32768;

      var src = ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate;
      src.connect(ctx.destination);

      var when = Math.max(playHead, ctx.currentTime + 0.02);
      src.start(when);
      sources.push(src);
      src.onended = (function (node) {
        return function () {
          var idx = sources.indexOf(node);
          if (idx >= 0) sources.splice(idx, 1);
        };
      })(src);

      playHead = when + buf.duration / rate;
      cursor = end;
    }

    if (!ended && streamDone && cursor >= totalSamples && ctx.currentTime >= playHead - 0.05) {
      ended = true;
      active = false;
      if (handlers.onEnd) handlers.onEnd();
    }
  }

  /* Restart scheduling from `sample`, discarding anything already queued.
     Used by seek and by rate changes, which both invalidate the queue. */
  function rescheduleFrom(sample) {
    if (!ctx) return;
    stopSources();
    cursor = Math.max(0, Math.min(Math.floor(sample), totalSamples));
    playHead = ctx.currentTime;
    ended = false;
    tick();
  }

  function play(query, minutes, h, context, voice) {
    handlers = h || {};
    stop();
    var myToken = ++token;
    active = true;

    ctx = new (window.AudioContext || window.webkitAudioContext)();
    controller = new AbortController();
    timer = setInterval(tick, 80);

    var url = "/api/audio?q=" + encodeURIComponent(query) +
              "&minutes=" + encodeURIComponent(minutes) + "&fmt=pcm" +
              (context ? "&context=" + encodeURIComponent(context) : "") +
              (voice ? "&voice=" + encodeURIComponent(voice) : "");

    ctx.resume().then(function () {
      return fetch(url, { signal: controller.signal });
    }).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () {
          return { error: "Request failed (" + res.status + ")" };
        }).then(function (body) { throw new Error(body.error || "Request failed"); });
      }
      sampleRate = Number(res.headers.get("X-Sample-Rate")) || 22050;

      var reader = res.body.getReader();
      var leftover = new Uint8Array(0);
      var received = 0;
      var first = true;

      function pump() {
        return reader.read().then(function (r) {
          if (myToken !== token) return;
          if (r.done) {
            streamDone = true;
            if (received === 0) throw new Error("The server sent an empty briefing.");
            return;
          }
          received += r.value.length;

          // A 16-bit sample can straddle a chunk boundary; carry the odd byte.
          var bytes = r.value;
          if (leftover.length) {
            var merged = new Uint8Array(leftover.length + bytes.length);
            merged.set(leftover, 0); merged.set(bytes, leftover.length);
            bytes = merged;
          }
          var usable = bytes.length - (bytes.length % 2);
          leftover = bytes.slice(usable);
          if (usable) {
            append(new Int16Array(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + usable)));
            if (first) {
              first = false;
              playHead = ctx.currentTime;
              tick();
              if (handlers.onFirstAudio) handlers.onFirstAudio();
            }
          }
          return pump();
        });
      }
      return pump();
    }).catch(function (err) {
      if (err && err.name === "AbortError") return;
      if (myToken !== token) return;
      active = false;
      if (handlers.onError) handlers.onError(err);
    });
  }

  function pause() { if (ctx && ctx.state === "running") ctx.suspend(); }
  function resume() { if (ctx && ctx.state === "suspended") ctx.resume(); }
  function isPaused() { return !!ctx && ctx.state === "suspended"; }

  function stop() { token++; reset(); }

  /* The furthest point that can be played right now. Once the whole episode
     has arrived that is its end; while it is still streaming, stop short so
     there is always audio left to keep playing. */
  function seekLimit() {
    var have = totalSamples / sampleRate;
    return streamDone ? have : Math.max(0, have - TAIL_MARGIN);
  }

  function seek(seconds) {
    if (!ctx || !totalSamples) return 0;
    var target = Math.max(0, Math.min(seconds, seekLimit()));
    rescheduleFrom(target * sampleRate);
    return cursor / sampleRate;
  }

  function skip(seconds) {
    if (!ctx || !totalSamples) return 0;
    return seek(positionSamples() / sampleRate + seconds);
  }

  function setRate(multiplier) {
    multiplier = Math.max(0.5, Math.min(3, Number(multiplier) || 1));
    if (!ctx) { rate = multiplier; return; }
    var here = positionSamples();
    rate = multiplier;
    rescheduleFrom(here);
  }

  return {
    play: play,
    pause: pause,
    resume: resume,
    isPaused: isPaused,
    stop: stop,
    skip: skip,
    seek: seek,
    setRate: setRate,
    // How far the listener may currently skip to, in seconds.
    seekLimit: seekLimit,
    getRate: function () { return rate; },
    position: function () { return positionSamples() / sampleRate; },
    // Seconds of audio received so far. Grows while the episode streams.
    duration: function () { return totalSamples / sampleRate; },
    isActive: function () { return active; },
    // True once the whole episode has been received.
    isComplete: function () { return streamDone; },
  };
})();
