import SwiftUI

/// The player, with both settled gestures.
///
/// The fifteen-second buttons and the drag answer different questions — the
/// buttons "say that again", the drag "get me to roughly there" — so neither
/// replaces the other and removing either would be a regression.
struct PlayerView: View {

    @ObservedObject var episode: EpisodeModel
    let onClose: () -> Void

    /// While dragging, the bar follows the finger rather than the audio clock;
    /// the seek lands on release.
    @State private var scrubbing: Double?

    var body: some View {
        VStack(spacing: 22) {
            Spacer(minLength: 8)

            Text(episode.title)
                .font(.title2.weight(.medium))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 8)

            progress

            controls

            HStack(spacing: 14) {
                Text("Speed")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Picker("Speed", selection: $episode.rate) {
                    Text("1×").tag(Float(1.0))
                    Text("1.25×").tag(Float(1.25))
                    Text("1.5×").tag(Float(1.5))
                    Text("2×").tag(Float(2.0))
                }
                .pickerStyle(.segmented)
            }

            if let next = episode.nextUp, let question = next.question {
                goDeeper(question)
            }

            Spacer()

            Button("Done", action: onClose)
                .padding(.bottom, 20)
        }
        .padding(20)
    }

    private var progress: some View {
        VStack(spacing: 6) {
            Slider(
                value: Binding(
                    get: { scrubbing ?? min(episode.position, max(episode.seekLimit, 0.01)) },
                    set: { scrubbing = $0 }
                ),
                // The bar spans what has actually been written. Dragging past
                // the edge would starve the player, so there is nothing there
                // to drag to.
                in: 0...max(episode.seekLimit, 0.01),
                onEditingChanged: { editing in
                    if !editing, let target = scrubbing {
                        episode.seek(to: target)
                        scrubbing = nil
                    }
                }
            )
            .disabled(episode.seekLimit <= 0)

            HStack {
                Text(clock(scrubbing ?? episode.position))
                Spacer()
                // What has arrived, not what was asked for — the episode is
                // still being written while it plays.
                Text(episode.isComplete ? clock(episode.received)
                                        : "\(clock(episode.received)) so far")
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }

    private var controls: some View {
        HStack(spacing: 34) {
            Button {
                episode.skip(-15)
            } label: {
                Image(systemName: "gobackward.15").font(.title)
            }
            .accessibilityLabel("Back fifteen seconds")

            Button {
                episode.togglePlayPause()
            } label: {
                Image(systemName: episode.isPaused ? "play.circle.fill" : "pause.circle.fill")
                    .font(.system(size: 62))
            }
            .accessibilityLabel(episode.isPaused ? "Play" : "Pause")

            Button {
                episode.skip(15)
            } label: {
                Image(systemName: "goforward.15").font(.title)
            }
            .accessibilityLabel("Forward fifteen seconds")
        }
    }

    /// The thread the episode left open, as one tap. Wanting to go deeper and
    /// actually doing it are separated by having to phrase a question; this
    /// removes that step.
    private func goDeeper(_ question: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("GO DEEPER")
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
            Text(question)
                .font(.subheadline.weight(.medium))
                .multilineTextAlignment(.leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .onTapGesture {
            episode.play(question: question, minutes: 3)
        }
    }

    private func clock(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
