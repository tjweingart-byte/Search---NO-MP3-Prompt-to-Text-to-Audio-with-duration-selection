import Foundation
import Security

/// The listener's session token, in the Keychain.
///
/// The settled rule is untouched and is the reason this type exists:
/// **a listener id is never accepted from the client.** The token here is the
/// same server-minted, high-entropy, revocable string the browser's HttpOnly
/// cookie carries — only the envelope changes, because `URLSession` keeps
/// cookies in `HTTPCookieStorage`, which iOS clears under conditions the app
/// does not control, and "the listener silently became a different listener" is
/// the worst possible failure for a product whose personalisation is an
/// append-only log keyed on that id.
///
/// `?user=` is never sent. There is no code path here that could.
///
/// The Keychain rather than `UserDefaults`: defaults are in a plist inside the
/// app container, readable from a backup. `kSecAttrAccessibleAfterFirstUnlock`
/// so an episode can keep streaming with the phone locked — the whole point of
/// the app — while still requiring one unlock since boot.
struct SessionStore {

    private let service = "com.fam.session"
    private let account = "listener-token"

    var token: String? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8),
              !value.isEmpty
        else { return nil }
        return value
    }

    /// Store a token the *server* minted. Nothing else may call this with a
    /// value it invented.
    func save(_ token: String) {
        let data = Data(token.utf8)
        var query = baseQuery()

        if SecItemCopyMatching(query as CFDictionary, nil) == errSecSuccess {
            let update: [String: Any] = [kSecValueData as String: data]
            SecItemUpdate(query as CFDictionary, update as CFDictionary)
            return
        }
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(query as CFDictionary, nil)
    }

    /// Sign out, and account deletion. After this the app is an anonymous
    /// listener again, which is a full identity — listening never required one.
    func clear() {
        SecItemDelete(baseQuery() as CFDictionary)
    }

    private func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }
}
