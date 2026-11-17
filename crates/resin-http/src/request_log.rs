//! The frontend's request log, on the `wright.http` logger: a line when a request arrives, is
//! queued or is rejected at info, and its headers and normalized body at debug. A request's
//! name in the log is its `x-request-id` header when the client sent one, else the id the
//! response carries; the response's `x-request-id` header repeats it either way.
use axum::{
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
};
use serde::Serialize;

use crate::{AppState, protocol::ConversationRequest};

pub(crate) const TARGET: &str = "wright.http";
const REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_CLIENT_ID_LENGTH: usize = 128;
/// Longer bodies (images in messages) are cut in the debug log.
const MAX_LOGGED_BODY: usize = 64 * 1024;

/// One request's identities for the log and the response, taken as it arrives.
pub(crate) struct RequestReceipt {
    /// The generated id, which the response body carries.
    pub(crate) id: String,
    /// The client's `x-request-id`, when it sent a usable one.
    pub(crate) client_request_id: Option<String>,
    route: &'static str,
}

impl RequestReceipt {
    pub(crate) fn new(
        state: &AppState,
        headers: &HeaderMap,
        prefix: &str,
        route: &'static str,
    ) -> Self {
        let receipt = Self {
            id: state.request_id(prefix),
            client_request_id: client_request_id(headers),
            route,
        };
        log::info!(target: TARGET, "{}: {route} received", receipt.log_id());
        if log::log_enabled!(target: TARGET, log::Level::Debug) {
            log::debug!(
                target: TARGET,
                "{}: headers {}",
                receipt.log_id(),
                describe_headers(headers)
            );
        }
        receipt
    }

    pub(crate) fn log_id(&self) -> &str {
        self.client_request_id.as_deref().unwrap_or(&self.id)
    }

    /// The request was not admitted to the queue: log the status, name the response.
    pub(crate) fn reject(&self, response: Response) -> Response {
        let status = response.status();
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            log::warn!(target: TARGET, "{}: {} rejected: HTTP {status}", self.log_id(), self.route);
        } else {
            log::info!(target: TARGET, "{}: {} rejected: HTTP {status}", self.log_id(), self.route);
        }
        self.respond(response)
    }

    /// The request waits for the inference actor.
    pub(crate) fn queued(&self, state: &AppState, request: &ConversationRequest, stream: bool) {
        let waiting = state
            .jobs
            .max_capacity()
            .saturating_sub(state.jobs.capacity());
        log::info!(
            target: TARGET,
            "{}: queued behind {} requests: {} messages, {} tools, max_output_tokens {}, stream {stream}",
            self.log_id(),
            waiting.saturating_sub(1),
            request.messages.len(),
            request.tools.len(),
            request
                .max_output_tokens
                .map_or_else(|| "unset".to_owned(), |limit| limit.to_string()),
        );
        if log::log_enabled!(target: TARGET, log::Level::Debug) {
            log::debug!(target: TARGET, "{}: request {}", self.log_id(), describe_body(request));
        }
    }

    /// Name the request in the response's `x-request-id` header.
    pub(crate) fn respond(&self, mut response: Response) -> Response {
        if let Ok(value) = HeaderValue::from_str(self.log_id()) {
            response.headers_mut().insert(REQUEST_ID_HEADER, value);
        }
        response
    }
}

/// A client's request id: visible ASCII of a bounded length, else none.
fn client_request_id(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(REQUEST_ID_HEADER)?.to_str().ok()?.trim();
    (!value.is_empty()
        && value.len() <= MAX_CLIENT_ID_LENGTH
        && value.bytes().all(|byte| (0x21..0x7f).contains(&byte)))
    .then(|| value.to_owned())
}

/// Every header, credentials replaced.
fn describe_headers(headers: &HeaderMap) -> String {
    let fields = headers
        .iter()
        .map(|(name, value)| {
            let shown = if matches!(
                *name,
                header::AUTHORIZATION | header::PROXY_AUTHORIZATION | header::COOKIE
            ) || name.as_str() == "x-api-key"
            {
                "[redacted]".to_owned()
            } else {
                value
                    .to_str()
                    .map_or_else(|_| format!("{value:?}"), str::to_owned)
            };
            format!("{name}: {shown}")
        })
        .collect::<Vec<_>>();
    format!("{{{}}}", fields.join(", "))
}

/// The normalized request as JSON, cut after `MAX_LOGGED_BODY` bytes.
fn describe_body(request: &impl Serialize) -> String {
    let mut body = serde_json::to_string(request).unwrap_or_else(|error| error.to_string());
    if body.len() > MAX_LOGGED_BODY {
        let total = body.len();
        let mut end = MAX_LOGGED_BODY;
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        body.truncate(end);
        body.push_str(&format!("... ({total} bytes)"));
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_request_id_accepts_visible_ascii_only() {
        let mut headers = HeaderMap::new();
        assert_eq!(client_request_id(&headers), None);
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static(" req-1 "));
        assert_eq!(client_request_id(&headers).as_deref(), Some("req-1"));
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static("has space"));
        assert_eq!(client_request_id(&headers), None);
        headers.insert(
            REQUEST_ID_HEADER,
            HeaderValue::from_str(&"x".repeat(MAX_CLIENT_ID_LENGTH + 1)).unwrap(),
        );
        assert_eq!(client_request_id(&headers), None);
    }

    #[test]
    fn headers_hide_credentials() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        headers.insert("x-api-key", HeaderValue::from_static("secret"));
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let shown = describe_headers(&headers);
        assert!(!shown.contains("secret"), "{shown}");
        assert!(shown.contains("authorization: [redacted]"), "{shown}");
        assert!(shown.contains("content-type: application/json"), "{shown}");
    }

    #[test]
    fn long_bodies_are_cut_at_a_character_boundary() {
        let body = describe_body(&"é".repeat(MAX_LOGGED_BODY));
        assert!(body.len() < MAX_LOGGED_BODY + 32);
        assert!(body.ends_with(" bytes)"), "{body}");
    }
}
