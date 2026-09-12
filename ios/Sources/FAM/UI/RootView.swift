import SwiftUI

/// v1 is search and the player, deliberately.
///
/// Explore replays episodes other listeners generated and echoes carry names,
/// which is squarely the user-generated-content rule however the text was
/// written. Leaving it out of the first build is what keeps the first review —
/// the one most likely to be rejected — a much smaller argument. myFAM,
/// DailyFAM and Explore all have endpoints already and are v1.1.
struct RootView: View {

    @StateObject private var episode = EpisodeModel()
    @State private var question = ""
    @State private var minutes = 3
    @State private var showDebug = false

    var body: some View {
        NavigationStack {
            Group {
                switch episode.phase {
                case .idle:
                    SearchView(question: $question, minutes: $minutes) {
                        episode.play(question: question, minutes: minutes)
                    }
                case let .waiting(since):
                    WaitingView(question: question, since: since) {
                        episode.stop()
                    }
                case .playing:
                    PlayerView(episode: episode) {
                        episode.stop()
                    }
                case let .failed(message):
                    FailureView(message: message) {
                        episode.stop()
                    }
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showDebug = true
                    } label: {
                        Image(systemName: "stethoscope")
                    }
                    .accessibilityLabel("Server status")
                }
            }
            .sheet(isPresented: $showDebug) { DebugView() }
        }
    }
}

/// Type a question, pick a length. The length is a ceiling, not a quota: an
/// episode that runs out of substance ends early rather than being padded.
struct SearchView: View {
    @Binding var question: String
    @Binding var minutes: Int
    let onGo: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("What do you want to know?")
                .font(.largeTitle.weight(.medium))
                .padding(.top, 12)

            TextField("Ask anything", text: $question, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(3...6)
                .padding(14)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
                .submitLabel(.go)
                .onSubmit(onGo)

            Picker("Length", selection: $minutes) {
                ForEach([1, 3, 5, 10], id: \.self) { value in
                    Text("\(value) min").tag(value)
                }
            }
            .pickerStyle(.segmented)

            Button(action: onGo) {
                Text("Listen")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .disabled(question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            Spacer()
        }
        .padding(20)
        .navigationTitle("FAM")
    }
}

/// An honest wait that names what it is waiting for and counts the seconds.
/// Nothing plays until the real briefing does.
struct WaitingView: View {
    let question: String
    let since: Date
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
                .controlSize(.large)
            Text(question)
                .font(.title3.weight(.medium))
                .multilineTextAlignment(.center)
            // Named, not disguised. The seconds are shown because a wait you
            // were warned about is a different experience from the same wait
            // unexplained.
            TimelineView(.periodic(from: since, by: 0.5)) { context in
                Text("Writing the episode · \(Int(context.date.timeIntervalSince(since)))s")
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Cancel", action: onCancel)
                .padding(.bottom, 24)
        }
        .padding(24)
    }
}

/// Every failure is a sentence the listener can act on.
struct FailureView: View {
    let message: String
    let onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.orange)
            Text(message)
                .multilineTextAlignment(.center)
                .font(.body)
            Spacer()
            Button("Back", action: onDismiss)
                .buttonStyle(.borderedProminent)
                .padding(.bottom, 24)
        }
        .padding(24)
    }
}
