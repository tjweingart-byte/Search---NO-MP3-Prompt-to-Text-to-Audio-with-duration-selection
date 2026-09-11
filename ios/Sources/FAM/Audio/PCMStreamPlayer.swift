import AVFoundation
import Foundation

/// Streaming speech with transport controls — the iOS half of `static/fam-audio.js`.
///
/// That file is the specification, not a file to translate line by line, and
/// its three load-bearing decisions are kept here exactly because each was
/// paid for in bugs:
///
///  * **every sample is retained as `Int16`**, because you cannot seek
///    backwards through audio you threw away. ~2.6 MB per minute, converted to
///    float only for the short slice being scheduled;
///  * **the position cursor is derived from the audio clock**, never from wall
///    time, so it stays correct across pauses and interruptions;
///  * **`tailMargin`** — while the episode is still being written, never seek
///    closer than two seconds to what has arrived. Landing on the edge starves
///    the player: nothing is left to schedule, playback stops dead, and further
///    skips appear to do nothing because the cursor is already pinned there.
///
/// What the browser did for free and iOS will not is deliberately *not* here:
/// the audio session, the lock screen and interruption handling belong to
/// `AudioSessionController` and `NowPlayingController`, so this type stays the
/// thing the spec describes and nothing else.
///
/// The architecture matches the Web Audio version — a node fed with scheduled
/// PCM buffers — which is why this is a port and not a redesign.
final class PCMStreamPlayer {

    // MARK: Tunables (the spec's, unchanged)

    /// Slice length scheduled at a time. Short slices keep seek and rate
    /// changes responsive, because a seek discards whatever is queued.
    private static let sliceSeconds = 0.25

    /// How far ahead of the clock to keep audio scheduled.
    private static let lookaheadSeconds = 0.35

    /// Never seek within this of the end of what has arrived. See the note above.
    static let tailMargin = 1.0 * 2.0

    // MARK: Engine

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    /// Time-pitch rather than varispeed: at 1.5x a varispeed voice is a
    /// chipmunk, and speed on spoken word is expected to preserve pitch.
    private let timePitch = AVAudioUnitTimePitch()

    // MARK: Retained audio

    /// Every sample received, as `Int16`. Grown, never discarded.
    private var pcm = [Int16]()
    /// Next sample to schedule.
    private var cursor: AVAudioFramePosition = 0
    /// Where the current playback run began, in absolute samples. `playerTime`
    /// restarts at zero after every `stop()`, so seeking needs this base.
    private var runBaseSample: AVAudioFramePosition = 0

    private var sampleRate: Double = 22_050
    private var streamDone = false
    private var ended = false
    private(set) var isActive = false
    private var isPausedFlag = false

    /// One queue owns every mutation of `pcm`, `cursor` and the engine, so the
    /// network callback and the transport controls cannot interleave.
    private let queue = DispatchQueue(label: "fam.audio.player")

    private var format: AVAudioFormat?
    private var stream: AudioStream?
    private var token = 0

    // MARK: Handlers

    var onFirstAudio: (() -> Void)?
    var onEnd: (() -> Void)?
    var onError: ((Error) -> Void)?
    /// Fired as audio arrives, so Now Playing can revise a duration that is
    /// still growing. Seconds received, and whether that is now the whole thing.
    var onProgress: ((_ received: Double, _ complete: Bool) -> Void)?

    // MARK: Lifecycle

    init() {
        engine.attach(player)
        engine.attach(timePitch)
    }

    /// Stream an episode and start speaking as soon as the first samples land.
    func play(request: EpisodeRequest, client: APIClient) {
        stop()
        queue.async { [self] in
            token += 1
            let myToken = token
            isActive = true
            ended = false
            streamDone = false
            isPausedFlag = false
            pcm.removeAll(keepingCapacity: true)
            cursor = 0
            runBaseSample = 0

            var started = false
            let stream = AudioStream()
            self.stream = stream

            stream.onHeaders = { [weak self] rate in
                self?.queue.async {
                    guard let self, myToken == self.token else { return }
                    // The server states the rate per stream; the 22050 here is
                    // only the fallback the spec names.
                    self.sampleRate = rate ?? 22_050
                }
            }
            stream.onSamples = { [weak self] samples in
                self?.queue.async {
                    guard let self, myToken == self.token else { return }
                    self.pcm.append(contentsOf: samples)
                    if !started {
                        started = true
                        do {
                            try self.startEngine()
                        } catch {
                            self.isActive = false
                            self.onError?(error)
                            return
                        }
                        self.onFirstAudio?()
                    }
                    self.pump()
                    self.onProgress?(self.receivedSeconds, false)
                }
            }
            stream.onFinished = { [weak self] error in
                self?.queue.async {
                    guard let self, myToken == self.token else { return }
                    if let error {
                        self.isActive = false
                        self.onError?(error)
                        return
                    }
                    self.streamDone = true
                    if self.pcm.isEmpty {
                        self.isActive = false
                        self.onError?(FAMError.emptyEpisode)
                        return
                    }
                    self.pump()
                    self.onProgress?(self.receivedSeconds, true)
                }
            }

            stream.start(url: client.audioURL(for: request), headers: client.authHeaders())
        }
    }

    func pause() {
        queue.async { [self] in
            guard isActive, !isPausedFlag else { return }
            isPausedFlag = true
            player.pause()
        }
    }

    func resume() {
        queue.async { [self] in
            guard isActive, isPausedFlag else { return }
            isPausedFlag = false
            player.play()
            pump()
        }
    }

    var isPaused: Bool { queue.sync { isPausedFlag } }

    func stop() {
        queue.async { [self] in
            token += 1
            stream?.cancel()
            stream = nil
            player.stop()
            if engine.isRunning { engine.stop() }
            pcm.removeAll(keepingCapacity: false)
            cursor = 0
            runBaseSample = 0
            isActive = false
            isPausedFlag = false
            streamDone = false
            ended = false
        }
    }

    // MARK: Transport

    /// Seconds of audio received so far. Grows while the episode streams.
    var receivedSeconds: Double {
        Double(pcm.count) / sampleRate
    }

    /// The furthest point playable right now. Once the whole episode has
    /// arrived that is its end; while it is still streaming, stop short, so
    /// there is always audio left to keep playing.
    var seekLimit: Double {
        queue.sync { seekLimitLocked }
    }

    private var seekLimitLocked: Double {
        let have = Double(pcm.count) / sampleRate
        return streamDone ? have : max(0, have - Self.tailMargin)
    }

    /// Current playback position in seconds, derived from the audio clock.
    var position: Double {
        queue.sync { positionLocked }
    }

    private var positionLocked: Double {
        guard let nodeTime = player.lastRenderTime,
              let playerTime = player.playerTime(forNodeTime: nodeTime) else {
            return Double(runBaseSample) / sampleRate
        }
        // `sampleTime` counts source samples this run has rendered, so it is
        // already rate-independent: the time-pitch unit downstream changes how
        // fast they are consumed, not what they mean.
        let played = max(0, playerTime.sampleTime)
        return Double(runBaseSample + played) / sampleRate
    }

    var duration: Double { queue.sync { Double(pcm.count) / sampleRate } }
    var isComplete: Bool { queue.sync { streamDone } }

    @discardableResult
    func seek(to seconds: Double) -> Double {
        queue.sync {
            guard !pcm.isEmpty else { return 0 }
            let target = min(max(0, seconds), seekLimitLocked)
            reschedule(from: AVAudioFramePosition(target * sampleRate))
            return Double(runBaseSample) / sampleRate
        }
    }

    @discardableResult
    func skip(_ seconds: Double) -> Double {
        let here = queue.sync { positionLocked }
        return seek(to: here + seconds)
    }

    /// 1 = normal. Clamped to the same range the browser build allows.
    func setRate(_ multiplier: Float) {
        queue.async { [self] in
            timePitch.rate = min(max(0.5, multiplier), 3.0)
        }
    }

    var rate: Float { timePitch.rate }

    // MARK: Engine plumbing

    private func startEngine() throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { throw FAMError.audioFormat }
        self.format = format

        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        player.play()
    }

    /// Keep the node fed. Called whenever new audio arrives, a buffer finishes,
    /// or the cursor moves — so it also picks up samples that landed after the
    /// queue had run dry.
    private func pump() {
        guard isActive, !isPausedFlag, let format else { return }

        let sliceFrames = AVAudioFrameCount(Self.sliceSeconds * sampleRate)
        let lookaheadSamples = AVAudioFramePosition(Self.lookaheadSeconds * sampleRate)
        let playedTo = AVAudioFramePosition(positionLocked * sampleRate)

        while cursor < AVAudioFramePosition(pcm.count),
              cursor - playedTo < lookaheadSamples {
            let end = min(cursor + AVAudioFramePosition(sliceFrames),
                          AVAudioFramePosition(pcm.count))
            let length = AVAudioFrameCount(end - cursor)
            guard length > 0,
                  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: length),
                  let channel = buffer.floatChannelData?[0] else { break }

            buffer.frameLength = length
            let base = Int(cursor)
            for i in 0..<Int(length) {
                channel[i] = Float(pcm[base + i]) / 32_768.0
            }

            player.scheduleBuffer(buffer) { [weak self] in
                // A finished buffer is the signal to schedule the next slice.
                self?.queue.async { self?.pump() }
            }
            cursor = end
        }

        if !ended, streamDone, cursor >= AVAudioFramePosition(pcm.count) {
            ended = true
            isActive = false
            onEnd?()
        }
    }

    /// Restart scheduling from `sample`, discarding whatever is queued. Used by
    /// seek, which invalidates the queue by definition.
    private func reschedule(from sample: AVAudioFramePosition) {
        let clamped = min(max(0, sample), AVAudioFramePosition(pcm.count))
        player.stop()                 // clears every scheduled buffer
        cursor = clamped
        runBaseSample = clamped       // playerTime restarts at zero from here
        ended = false
        if !isPausedFlag { player.play() }
        pump()
    }
}

enum FAMError: LocalizedError {
    case emptyEpisode
    case audioFormat
    case http(Int, String)

    var errorDescription: String? {
        switch self {
        case .emptyEpisode:
            return "The server sent an empty briefing."
        case .audioFormat:
            return "This device could not open an audio channel for the episode."
        case let .http(status, message):
            return message.isEmpty ? "Request failed (\(status))" : message
        }
    }
}
