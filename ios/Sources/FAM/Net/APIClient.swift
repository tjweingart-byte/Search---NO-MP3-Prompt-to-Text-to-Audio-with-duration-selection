import Foundation

/// What the listener asked for. Mirrors `/api/audio`'s parameters exactly;
/// `fmt` is always `pcm`, because this client schedules samples itself.
struct EpisodeRequest {
    var query: String
    var minutes: Int = 3
    /// Topic the listener just heard, for a follow-up.
    var context: String = ""
    /// Voice id from `/api/voices`.
    var voice: String = ""
    /// Bank topic id, when played from a tile, so the feed can rank.
    var topicID: String = ""
    /// Explore replays only. The server refuses to generate on a miss, so a
    /// stale card costs a 409 rather than a model call.
    var cachedOnly: Bool = false
}

/// The one place that knows the server's shape.
///
/// Everything goes through `/api/v1`. That prefix exists precisely for this
/// client: an app on somebody's phone cannot be redeployed with the server, so
/// the day an endpoint changes shape, `/api/v2` can carry the new one while
/// `/api/v1` keeps the promise made to every copy already installed.
struct APIClient {

    let baseURL: URL
    let session: SessionStore

    init(baseURL: URL = Config.apiBaseURL, session: SessionStore = SessionStore()) {
        self.baseURL = baseURL
        self.session = session
    }

    private var v1: URL { baseURL.appendingPathComponent("api/v1") }

    /// Headers every request carries. The bearer token is the cookie's twin —
    /// see `SessionStore`. A browser must never ask for one; this is not a
    /// browser.
    func authHeaders() -> [String: String] {
        var headers = ["Accept": "*/*"]
        if let token = session.token {
            headers["Authorization"] = "Bearer \(token)"
        }
        return headers
    }

    // MARK: The audio path

    func audioURL(for request: EpisodeRequest) -> URL {
        var components = URLComponents(
            url: v1.appendingPathComponent("audio"),
            resolvingAgainstBaseURL: false)!

        var items = [
            URLQueryItem(name: "q", value: request.query),
            URLQueryItem(name: "minutes", value: String(request.minutes)),
            // Bare samples. `fmt=wav` exists for a plain <audio> tag and is not
            // what this client wants.
            URLQueryItem(name: "fmt", value: "pcm"),
        ]
        if !request.context.isEmpty {
            items.append(URLQueryItem(name: "context", value: request.context))
        }
        if !request.voice.isEmpty {
            items.append(URLQueryItem(name: "voice", value: request.voice))
        }
        if !request.topicID.isEmpty {
            items.append(URLQueryItem(name: "topic_id", value: request.topicID))
        }
        if request.cachedOnly {
            items.append(URLQueryItem(name: "cached_only", value: "true"))
        }
        // Note what is *not* here and never will be: `user=`. Who is listening
        // comes from the session, which the server minted.
        components.queryItems = items
        return components.url!
    }

    // MARK: JSON

    /// Ask the server for the predicted follow-up this episode left standing.
    /// Free — the pipeline stored it beside the script — so Go Deeper costs
    /// nothing to anyone who does not tap it.
    func next(after query: String) async throws -> NextSuggestion? {
        var components = URLComponents(
            url: v1.appendingPathComponent("next"),
            resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "q", value: query)]
        return try? await get(components.url!, as: NextSuggestion.self)
    }

    /// Readiness, as the server actually measures it rather than as it was
    /// configured — "verify, do not inspect". Used by the debug screen so a
    /// tester can say *which* thing is wrong rather than "it didn't work".
    func health() async throws -> Health {
        try await get(baseURL.appendingPathComponent("api/health"), as: Health.self)
    }

    /// This listener's tier, limits and what is left.
    func entitlements() async throws -> Entitlements {
        try await get(v1.appendingPathComponent("entitlements"), as: Entitlements.self)
    }

    private func get<T: Decodable>(_ url: URL, as type: T.Type) async throws -> T {
        var request = URLRequest(url: url)
        for (key, value) in authHeaders() {
            request.setValue(value, forHTTPHeaderField: key)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            let message = (try? JSONDecoder().decode(ServerError.self, from: data))?.error ?? ""
            throw FAMError.http(http.statusCode, message)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}

/// The server's own sentence on a refusal. It uses `error` for its own
/// messages and `detail` for FastAPI's, so both are read — a refusal the
/// listener cannot act on is the failure this project keeps paying for.
private struct ServerError: Decodable {
    let error: String?

    enum CodingKeys: String, CodingKey { case error, detail }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        error = try container.decodeIfPresent(String.self, forKey: .error)
            ?? container.decodeIfPresent(String.self, forKey: .detail)
    }
}
