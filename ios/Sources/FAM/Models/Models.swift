import Foundation

/// The follow-up the episode left standing, predicted rather than promised.
/// The script is barred from gesturing at it; it waits here afterwards for
/// anyone who wants it.
struct NextSuggestion: Decodable {
    let question: String?
    let context: String?
}

/// `/api/health`, which reports what the machine will actually do.
///
/// `interim` is the one that matters on a tester's phone: true means nothing
/// can speak and playback is a **placeholder tone**, deliberately
/// indistinguishable from a broken app because that is what it is for. A tone
/// cannot be mistaken for FAM; a flat neural voice can. Surfacing this is why
/// the debug screen exists — "it didn't work" is not a bug report, and the
/// server already knows which of the four ways it is failing.
struct Health: Decodable {
    let interim: Bool?
    let engine: String?
    let voice: String?
    let cacheEntries: Int?

    enum CodingKeys: String, CodingKey {
        case interim, engine, voice
        case cacheEntries = "cache_entries"
    }

    /// True when this server will really speak. Anything else must say so out
    /// loud rather than sounding quietly worse than intended.
    var willSpeak: Bool { interim != true }
}

/// This listener's tier, limits and what is left.
///
/// A tier is what you may spend, never what you may reach: the free tier is a
/// daily ceiling on episodes, not a smaller product.
struct Entitlements: Decodable {
    let tier: String?
    let episodesRemaining: Int?
    let unlimited: Bool?

    enum CodingKeys: String, CodingKey {
        case tier, unlimited
        case episodesRemaining = "episodes_remaining"
    }
}
