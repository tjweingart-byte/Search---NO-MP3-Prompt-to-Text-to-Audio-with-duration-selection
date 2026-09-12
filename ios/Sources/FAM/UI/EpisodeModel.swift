import Foundation
import SwiftUI

/// What a screen needs to know about the episode that is playing.
///
/// Everything the interface shows about progress is derived from the player's
/// audio clock, never from a wall-clock timer, so a pause or an interruption
/// cannot make the bar and the sound disagree.
@MainActor
final class EpisodeModel: ObservableObject {

    enum Phase: Equatable {
        case idle
        /// Generating. The interface says what it is waiting for and counts the
        /// seconds — no filler, ever, and no setting for it. A wait you were
        /// warned about is a different experience from the same wait unexplained.
        case waiting(since: Date)
        case playing
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var title = ""
    @Published private(set) var position: Double = 0
    @Published private(set) var received: Double = 0
    @Published private(set) var seekLimit: Double = 0
    @Published private(set) var isPaused = false
    @Published private(set) var isComplete = false
    @Published var rate: Float = 1.0 { didSet { player.setRate(rate) } }
    /// The predicted follow-up, offered as one tap once the episode ends.
    @Published private(set) var nextUp: NextSuggestion?

    let player = PCMStreamPlayer()
    private lazy var nowPlaying = NowPlayingController(player: player)
    private lazy var audioSession = AudioSessionController(player: player)
    private let client = APIClient()
    private var ticker: Timer?
    private var currentQuery = ""

    init() {
        player.onFirstAudio = { [weak self] in
            Task { @MainActor in self?.phase = .playing }
        }
        player.onProgress = { [weak self] received, complete in
            Task { @MainActor in
                self?.received = received
                self?.isComplete = complete
                self?.nowPlaying.update()
            }
        }
        player.onEnd = { [weak self] in
            Task { @MainActor in await self?.finish() }
        }
        player.onError = { [weak self] error in
            Task { @MainActor in
                self?.phase = .failed(error.localizedDescription)
                self?.stopTicking()
            }
        }
    }

    func play(question: String, minutes: Int) {
        guard !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        do {
            try audioSession.activate()
        } catch {
            phase = .failed("This device would not open an audio session: \(error.localizedDescription)")
            return
        }

        currentQuery = question
        title = question
        nextUp = nil
        position = 0
        received = 0
        isComplete = false
        phase = .waiting(since: Date())

        player.play(request: EpisodeRequest(query: question, minutes: minutes), client: client)
        nowPlaying.start(title: question)
        startTicking()
    }

    func togglePlayPause() {
        isPaused ? player.resume() : player.pause()
        isPaused = player.isPaused
        nowPlaying.update()
    }

    func skip(_ seconds: Double) {
        player.skip(seconds)
        refresh()
        nowPlaying.update()
    }

    /// The drag. It clamps at what has actually been written, because the
    /// episode is still being generated while it plays.
    func seek(to seconds: Double) {
        player.seek(to: seconds)
        refresh()
        nowPlaying.update()
    }

    func stop() {
        player.stop()
        nowPlaying.stop()
        audioSession.deactivate()
        stopTicking()
        phase = .idle
    }

    private func finish() async {
        stopTicking()
        refresh()
        // Fetched only now, so it costs nothing to anyone who never reaches the
        // end of an episode.
        nextUp = try? await client.next(after: currentQuery)
    }

    private func startTicking() {
        stopTicking()
        // Only the interface refreshes on this timer; the position it reads is
        // still the audio clock's.
        ticker = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    private func stopTicking() {
        ticker?.invalidate()
        ticker = nil
    }

    private func refresh() {
        position = player.position
        seekLimit = player.seekLimit
        isPaused = player.isPaused
        isComplete = player.isComplete
    }
}
