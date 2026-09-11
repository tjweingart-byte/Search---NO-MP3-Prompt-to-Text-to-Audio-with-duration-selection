import MediaPlayer
import Foundation

/// The lock screen, Control Centre, CarPlay and the AirPods stem.
///
/// CLAUDE.md settles the transport as two gestures that answer different
/// questions — the fifteen-second buttons ("say that again") and the drag
/// ("get me to roughly there") — and says neither replaces the other. On a
/// phone they are also hardware, so both are wired here: `skipForward` /
/// `skipBackward` at fifteen seconds, and `changePlaybackPosition` for the
/// scrubber.
///
/// **Duration is unknown while streaming.** Now Playing wants a total and the
/// episode is still being written, so report what has arrived and keep
/// revising it — the same thing the in-app scrubber does. Claiming a final
/// duration early would make the lock-screen scrubber lie.
final class NowPlayingController {

    private let player: PCMStreamPlayer
    private var title: String = "FAM"

    init(player: PCMStreamPlayer) {
        self.player = player
    }

    func start(title: String) {
        self.title = title
        wireCommands()
        update()
    }

    func stop() {
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
    }

    /// Push the current position and the duration-so-far to the lock screen.
    func update() {
        var info = [String: Any]()
        info[MPMediaItemPropertyTitle] = title
        info[MPMediaItemPropertyArtist] = "FAM"
        // What has arrived, not what was asked for. Revised on every chunk.
        info[MPMediaItemPropertyPlaybackDuration] = player.duration
        info[MPNowPlayingInfoPropertyElapsedPlaybackTime] = player.position
        info[MPNowPlayingInfoPropertyPlaybackRate] = player.isPaused ? 0.0 : Double(player.rate)
        // Until the whole episode has arrived this really is a live stream, and
        // saying so stops the scrubber pretending it knows where the end is.
        info[MPNowPlayingInfoPropertyIsLiveStream] = !player.isComplete
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
    }

    private func wireCommands() {
        let centre = MPRemoteCommandCenter.shared()

        centre.playCommand.removeTarget(nil)
        centre.playCommand.addTarget { [weak self] _ in
            self?.player.resume(); self?.update(); return .success
        }

        centre.pauseCommand.removeTarget(nil)
        centre.pauseCommand.addTarget { [weak self] _ in
            self?.player.pause(); self?.update(); return .success
        }

        centre.togglePlayPauseCommand.removeTarget(nil)
        centre.togglePlayPauseCommand.addTarget { [weak self] _ in
            guard let self else { return .commandFailed }
            player.isPaused ? player.resume() : player.pause()
            update()
            return .success
        }

        // The settled fifteen seconds, now as hardware.
        centre.skipForwardCommand.preferredIntervals = [15]
        centre.skipForwardCommand.removeTarget(nil)
        centre.skipForwardCommand.addTarget { [weak self] _ in
            self?.player.skip(15); self?.update(); return .success
        }

        centre.skipBackwardCommand.preferredIntervals = [15]
        centre.skipBackwardCommand.removeTarget(nil)
        centre.skipBackwardCommand.addTarget { [weak self] _ in
            self?.player.skip(-15); self?.update(); return .success
        }

        // The drag. It clamps at what has actually been written, because the
        // episode is still being generated while it plays.
        centre.changePlaybackPositionCommand.removeTarget(nil)
        centre.changePlaybackPositionCommand.addTarget { [weak self] event in
            guard let self,
                  let event = event as? MPChangePlaybackPositionCommandEvent
            else { return .commandFailed }
            player.seek(to: event.positionTime)
            update()
            return .success
        }

        // Nothing here is a track list, so these would be dead buttons.
        centre.nextTrackCommand.isEnabled = false
        centre.previousTrackCommand.isEnabled = false
    }
}
