/*
 * FamAudio - streaming speech for the FAM prototype.
 *
 * Replaces window.speechSynthesis. Instead of handing a string to the OS voice,
 * this asks the server for a briefing and plays the raw sound samples as they
 * arrive, so audio starts long before the episode has finished being written.
 *
 * Nothing is downloaded as a file and nothing is encoded: the response is a
 * stream of 16-bit PCM that gets scheduled straight onto a Web Audio clock.
 *
 * Public surface, deliberately shaped like the speechSynthesis calls it
 * replaces so the prototype's own logic barely changes:
 *
 *   FamAudio.play(query, minutes, handlers)  -> begin a briefing
 *   FamAudio.pause() / resume() / stop()
 *   FamAudio.elapsed()                       -> seconds heard so far
 *   FamAudio.isActive()
 */
window.FamAudio = (function () {
  "use strict";

  var ctx = null;          // AudioContext
  var controller = null;   // aborts the in-flight request
  var playHead = 0;        // next scheduled start time, on the ctx clock
  var startedAt = 0;       // ctx time when the first sample was scheduled
  var active = false;
  var endTimer = null;
  var token = 0;           // guards against a stale stream finishing late

  function teardown() {
    if (endTimer) { clearTimeout(endTimer); endTimer = null; }
    if (controller) { try { controller.abort(); } catch (e) {} controller = null; }
    if (ctx) { var c = ctx; ctx = null; c.close().catch(function () {}); }
    active = false;
  }

  /* One chunk of PCM -> one scheduled buffer, appended to the play head. */
  function schedule(int16, sampleRate) {
    var buf = ctx.createBuffer(1, int16.length, sampleRate);
    var out = buf.getChannelData(0);
    for (var i = 0; i < int16.length; i++) out[i] = int16[i] / 32768;

    var src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);

    // Never schedule in the past: if the network stalled and the play head fell
    // behind the clock, restart from now rather than dropping the chunk.
    var when = Math.max(playHead, ctx.currentTime + 0.05);
    src.start(when);
    playHead = when + buf.duration;
  }

  function play(query, minutes, handlers) {
    handlers = handlers || {};
    stop();

    var myToken = ++token;
    active = true;

    // Must be constructed inside the user gesture or browsers keep it suspended.
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    playHead = 0;
    startedAt = 0;
    controller = new AbortController();

    var url = "/api/audio?q=" + encodeURIComponent(query) +
              "&minutes=" + encodeURIComponent(minutes) + "&fmt=pcm";

    ctx.resume().then(function () {
      return fetch(url, { signal: controller.signal });
    }).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () {
          return { error: "Request failed (" + res.status + ")" };
        }).then(function (body) {
          throw new Error(body.error || "Request failed (" + res.status + ")");
        });
      }

      var sampleRate = Number(res.headers.get("X-Sample-Rate")) || 22050;
      var reader = res.body.getReader();
      var leftover = new Uint8Array(0);
      var received = 0;
      var first = true;

      function pump() {
        return reader.read().then(function (result) {
          if (myToken !== token) return;            // superseded by a newer play()
          if (result.done) {
            if (received === 0) {
              throw new Error("The server sent an empty briefing. Check the server log.");
            }
            // Let the tail finish before declaring the episode over.
            var remaining = Math.max(0, playHead - ctx.currentTime) * 1000;
            endTimer = setTimeout(function () {
              if (myToken !== token) return;
              active = false;
              if (handlers.onEnd) handlers.onEnd();
            }, remaining + 120);
            return;
          }

          var value = result.value;
          received += value.length;

          // A 16-bit sample can straddle a chunk boundary; carry the odd byte.
          var bytes = value;
          if (leftover.length) {
            var merged = new Uint8Array(leftover.length + value.length);
            merged.set(leftover, 0);
            merged.set(value, leftover.length);
            bytes = merged;
          }
          var usable = bytes.length - (bytes.length % 2);
          leftover = bytes.slice(usable);

          if (usable) {
            schedule(
              new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + usable)),
              sampleRate
            );
            if (first) {
              first = false;
              startedAt = ctx.currentTime;
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

  /* Suspending the context freezes its clock, so elapsed() stays correct
     across a pause without any bookkeeping of our own. */
  function pause() { if (ctx && ctx.state === "running") ctx.suspend(); }
  function resume() { if (ctx && ctx.state === "suspended") ctx.resume(); }

  function stop() {
    token++;
    teardown();
  }

  function elapsed() {
    if (!ctx || !startedAt) return 0;
    return Math.max(0, ctx.currentTime - startedAt);
  }

  return {
    play: play,
    pause: pause,
    resume: resume,
    stop: stop,
    elapsed: elapsed,
    isActive: function () { return active; },
  };
})();
