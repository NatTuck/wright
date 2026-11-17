//! Several models behind one router: routing by the `model` field, the 404 of an unknown
//! model, `GET /v1/models`, and per-model frontend metrics.

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use resin_http::{
    FinishReason, GenerationEvent, GenerationJob, MetricsEndpoint, ModelMetadata, ModelRoute,
    RouteConfig, ServedModel, router_for,
};
use serde_json::{Value, json};
use std::num::NonZeroU32;
use tokio::sync::mpsc;
use tower::ServiceExt;

/// A queue whose actor answers every request with `text`.
fn model(id: &str, text: &'static str, context: u32) -> ModelRoute {
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(8);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens: 3 });
            let _ = job.events.try_send(GenerationEvent::FirstToken);
            let _ = job.events.try_send(GenerationEvent::TextDelta {
                delta: text.to_owned(),
                logprobs: vec![],
            });
            let _ = job.events.try_send(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                input_tokens: 3,
                cached_tokens: 0,
                output_tokens: 1,
                elapsed_ms: 1.0,
                generation_elapsed_ms: 1.0,
                time_to_first_token_ms: 1.0,
            });
        }
    });
    ModelRoute {
        model: ServedModel::new(
            id,
            ModelMetadata {
                max_context_tokens: NonZeroU32::new(context),
                ..ModelMetadata::default()
            },
        ),
        jobs,
    }
}

fn two_models() -> axum::Router {
    router_for(
        vec![
            model("big", "from big", 131072),
            model("small", "from small", 32768),
        ],
        RouteConfig::default(),
        None,
    )
}

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn requests_go_to_the_model_they_name() {
    let app = two_models();
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (id, text) in [("big", "from big"), ("small", "from small")] {
        let (status, body) = post(
            app.clone(),
            "/v1/chat/completions",
            json!({"model": id, "messages": messages}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], text);
        let (status, body) = post(
            app.clone(),
            "/v1/messages",
            json!({"model": id, "max_tokens": 8, "messages": messages}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["content"][0]["text"], text);
    }
}

#[tokio::test]
async fn an_unknown_model_gets_the_served_ids() {
    let app = two_models();
    let messages = json!([{"role": "user", "content": "hi"}]);
    let (status, body) = post(
        app.clone(),
        "/v1/chat/completions",
        json!({"model": "other", "messages": messages}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains(r#""big", "small""#), "{message}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let (status, body) = post(
        app,
        "/v1/messages",
        json!({"model": "other", "max_tokens": 8, "messages": messages}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn the_model_list_has_every_model_with_its_metadata() {
    let response = two_models()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["id"], "big");
    assert_eq!(data[0]["max_context_tokens"], 131072);
    assert_eq!(data[1]["id"], "small");
    assert_eq!(data[1]["max_context_tokens"], 32768);
}

#[tokio::test]
async fn health_needs_every_model() {
    let closed = {
        let (jobs, receiver) = mpsc::channel::<GenerationJob>(1);
        drop(receiver);
        ModelRoute {
            model: ServedModel::from("stopped"),
            jobs,
        }
    };
    let app = router_for(
        vec![model("running", "x", 16), closed],
        RouteConfig::default(),
        None,
    );
    let response = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
#[should_panic(expected = "served twice")]
fn a_model_id_is_served_once() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _guard = runtime.enter();
    let _ = router_for(
        vec![model("same", "a", 16), model("same", "b", 16)],
        RouteConfig::default(),
        None,
    );
}

#[tokio::test]
async fn each_model_has_its_own_frontend_metrics() {
    let first = MetricsEndpoint::new("big");
    let second = first.for_model("small");
    first.set_queue_size(1);
    second.set_queue_size(4);
    second.observe_completed(0.2);
    second.publish(Vec::new()).unwrap(); // one publication for both handles
    let app = router_for(
        vec![model("big", "x", 16), model("small", "y", 16)],
        RouteConfig::default(),
        Some(first),
    );
    let response = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    for (model, waiting) in [("big", 1), ("small", 4)] {
        let line = format!(
            r#"resin:http_num_queue_reqs{{model_name="{model}",engine_type="resin"}} {waiting}"#
        );
        assert!(text.contains(&line), "{line} not in {text}");
    }
}
