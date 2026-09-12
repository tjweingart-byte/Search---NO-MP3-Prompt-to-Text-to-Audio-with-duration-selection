import Foundation

/// Raw PCM off the wire, as it arrives.
///
/// Deliberately `URLSessionDataDelegate` rather than `URLSession.bytes`:
/// `AsyncBytes` yields one byte at a time, and an episode is 2.65 MB a minute.
/// The delegate hands over whole chunks, which is what the one-sentence spec's
/// half-second budget needs.
///
/// Two details carried straight from the browser build:
///
///  * **a 16-bit sample can straddle a chunk boundary**, so an odd trailing
///    byte is carried into the next chunk rather than dropped — dropping it
///    desynchronises every sample after it and the episode turns to noise;
///  * **the sample rate comes from `X-Sample-Rate`**, per stream, because the
///    engine states it and the client must not assume it.
///
/// No MP3, no audio file: this decodes nothing and writes nothing. The bytes
/// go to the player's buffer and nowhere else.
final class AudioStream: NSObject {

    var onHeaders: ((Double?) -> Void)?
    var onSamples: (([Int16]) -> Void)?
    var onFinished: ((Error?) -> Void)?

    private var session: URLSession?
    private var task: URLSessionDataTask?
    /// The odd byte carried across a chunk boundary.
    private var leftover = Data()
    private var received = 0
    private var failure: Error?

    func start(url: URL, headers: [String: String]) {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        // A stream has no useful timeout between bytes once it has started, but
        // it must not hang forever before the first one.
        request.timeoutInterval = 60
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }

        let configuration = URLSessionConfiguration.default
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.waitsForConnectivity = true
        let session = URLSession(configuration: configuration,
                                 delegate: self,
                                 delegateQueue: nil)
        self.session = session
        task = session.dataTask(with: request)
        task?.resume()
    }

    func cancel() {
        task?.cancel()
        session?.invalidateAndCancel()
        session = nil
        task = nil
    }
}

extension AudioStream: URLSessionDataDelegate {

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.allow); return
        }
        guard (200..<300).contains(http.statusCode) else {
            // The server puts a readable sentence in the body on refusal — a
            // quota verdict, a cache miss on a `cached_only` replay. Let the
            // body arrive so the listener is told which, then fail.
            failure = FAMError.http(http.statusCode, "")
            completionHandler(.allow)
            return
        }
        let rate = (http.value(forHTTPHeaderField: "X-Sample-Rate")).flatMap(Double.init)
        onHeaders?(rate)
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        if failure != nil {
            // Error body, not audio. Collect it for the message.
            leftover.append(data)
            return
        }

        received += data.count
        var bytes = data
        if !leftover.isEmpty {
            bytes = leftover + bytes
            leftover.removeAll(keepingCapacity: true)
        }

        // A 16-bit sample cannot be half-delivered; carry the odd byte.
        let usable = bytes.count - (bytes.count % 2)
        if usable < bytes.count {
            leftover = bytes.suffix(from: usable)
        }
        guard usable > 0 else { return }

        let samples: [Int16] = bytes.prefix(usable).withUnsafeBytes { raw in
            // The stream is little-endian 16-bit mono, and every platform this
            // ships on is little-endian, so this is a reinterpret, not a copy
            // per sample.
            Array(raw.bindMemory(to: Int16.self))
        }
        onSamples?(samples)
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        defer {
            self.session?.finishTasksAndInvalidate()
            self.session = nil
        }

        if let error = error as NSError?, error.code == NSURLErrorCancelled {
            return  // a deliberate stop is not a failure
        }
        if let failure = failure as? FAMError, case let .http(status, _) = failure {
            let message = Self.message(from: leftover)
            onFinished?(FAMError.http(status, message))
            return
        }
        onFinished?(error)
    }

    /// The server's own sentence, when it sent one.
    private static func message(from body: Data) -> String {
        guard !body.isEmpty,
              let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let detail = (object["error"] ?? object["detail"]) as? String
        else { return "" }
        return detail
    }
}
