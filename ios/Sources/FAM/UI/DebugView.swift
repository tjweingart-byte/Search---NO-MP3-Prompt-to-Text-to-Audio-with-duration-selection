import SwiftUI

/// What this build is talking to, and whether that server will actually speak.
///
/// This exists because of the failure this project has lost the most time to:
/// silent success. A build pointed at a server with no voice model plays a
/// **placeholder tone**, which is deliberately indistinguishable from a broken
/// app — so a tester reporting "it didn't work" is reporting four different
/// possible faults at once. The server already knows which; this asks it.
///
/// "Verify, do not inspect": `/api/health` performs the real checks rather
/// than confirming that something was configured.
struct DebugView: View {

    @Environment(\.dismiss) private var dismiss
    @State private var health: Health?
    @State private var entitlements: Entitlements?
    @State private var error: String?
    @State private var loading = true

    private let client = APIClient()

    var body: some View {
        NavigationStack {
            List {
                Section("This build") {
                    row("Version", Config.buildLabel)
                    row("Server", Config.apiBaseURL.absoluteString)
                }

                Section("Will it speak?") {
                    if loading {
                        HStack { ProgressView(); Text("Asking the server…") }
                    } else if let error {
                        Label(error, systemImage: "xmark.octagon")
                            .foregroundStyle(.red)
                    } else if let health {
                        if health.willSpeak {
                            Label("Chatterbox is speaking", systemImage: "checkmark.circle")
                                .foregroundStyle(.green)
                        } else {
                            // Announce it. An app that quietly sounds worse than
                            // intended is the failure being guarded against.
                            Label(
                                "No voice model on this server — you will hear a placeholder tone, not FAM.",
                                systemImage: "exclamationmark.triangle"
                            )
                            .foregroundStyle(.orange)
                        }
                        if let engine = health.engine { row("Engine", engine) }
                        if let voice = health.voice { row("Voice", voice) }
                        if let entries = health.cacheEntries {
                            row("Cached scripts", String(entries))
                        }
                    }
                }

                if let entitlements {
                    Section("This listener") {
                        row("Tier", entitlements.tier ?? "—")
                        if entitlements.unlimited == true {
                            row("Episodes left", "unlimited")
                        } else if let left = entitlements.episodesRemaining {
                            row("Episodes left", String(left))
                        }
                    }
                }
            }
            .navigationTitle("Server status")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await load() }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func load() async {
        defer { loading = false }
        do {
            health = try await client.health()
        } catch {
            self.error = error.localizedDescription
            return
        }
        entitlements = try? await client.entitlements()
    }
}
