import Foundation

/// Build-time configuration.
///
/// The API base URL comes from `Info.plist`, set per configuration in
/// `project.yml`, so a TestFlight build points at the deployed server and a
/// Debug build can point at a laptop on the same wifi — without a code change
/// and without a URL ever being typed into a source file.
///
/// This is the same reasoning as `~/.fam/` on the server side: a value that has
/// to be re-entered by hand is a value that will eventually be entered wrong.
enum Config {

    /// Where the API lives. Fails loudly rather than defaulting to something
    /// plausible — a build pointed at the wrong server is exactly the quiet
    /// failure this project has lost the most time to.
    static var apiBaseURL: URL {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: "FAMAPIBaseURL") as? String,
              !raw.isEmpty,
              let url = URL(string: raw) else {
            fatalError("""
                FAMAPIBaseURL is missing from Info.plist. Set API_BASE_URL in \
                project.yml (or pass it to xcodegen) and regenerate the project.
                """)
        }
        return url
    }

    /// Shown on the debug screen so a tester can report which build they have.
    static var buildLabel: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        return "\(version) (\(build))"
    }
}
