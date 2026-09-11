import AVFoundation
import Foundation

/// The audio session, and the two events that do not exist in a browser build
/// and are ordinary on a phone: a call arriving, and AirPods being pulled out.
///
/// This is half the reason the app is not a web view. `AVAudioSession` category
/// `.playback` plus the `audio` background mode is what keeps an episode
/// speaking once the phone locks; a `WKWebView`'s `AudioContext` is suspended
/// at exactly that moment, and no configuration changes it.
final class AudioSessionController {

    private let player: PCMStreamPlayer
    /// Whether *we* paused for an interruption, as opposed to the listener
    /// pausing. Only the first kind should resume itself.
    private var pausedByInterruption = false

    init(player: PCMStreamPlayer) {
        self.player = player
    }

    /// Call once, before the first episode.
    func activate() throws {
        let session = AVAudioSession.sharedInstance()
        // `.spokenAudio` tells the system this is speech, not music: it is what
        // makes "pause other audio" and car integrations behave sensibly.
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true)
        observe()
    }

    func deactivate() {
        try? AVAudioSession.sharedInstance().setActive(
            false, options: .notifyOthersOnDeactivation)
    }

    private func observe() {
        let centre = NotificationCenter.default
        centre.addObserver(
            self, selector: #selector(handleInterruption(_:)),
            name: AVAudioSession.interruptionNotification, object: nil)
        centre.addObserver(
            self, selector: #selector(handleRouteChange(_:)),
            name: AVAudioSession.routeChangeNotification, object: nil)
    }

    // MARK: Interruptions — a phone call, a timer, Siri

    @objc private func handleInterruption(_ note: Notification) {
        guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }

        switch type {
        case .began:
            if !player.isPaused {
                pausedByInterruption = true
                player.pause()
            }
        case .ended:
            guard pausedByInterruption else { return }
            pausedByInterruption = false
            // Resume only when the system says we may. Anything else is an app
            // that starts talking over whatever interrupted it.
            let options = (note.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt)
                .map(AVAudioSession.InterruptionOptions.init(rawValue:)) ?? []
            if options.contains(.shouldResume) {
                try? AVAudioSession.sharedInstance().setActive(true)
                player.resume()
            }
        @unknown default:
            break
        }
    }

    // MARK: Route changes — headphones out

    @objc private func handleRouteChange(_ note: Notification) {
        guard let raw = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: raw) else { return }

        // The only case that must be handled: the thing playing the audio was
        // removed. Continuing would play the episode out loud in a room the
        // listener chose headphones for.
        if reason == .oldDeviceUnavailable {
            player.pause()
        }
    }
}
