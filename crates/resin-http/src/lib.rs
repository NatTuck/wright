mod anthropic;
mod metrics;
mod metrics_encoder;
mod openai;
mod protocol;
mod request_log;
mod ui;

use std::{
    future::{Future, IntoFuture},
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

pub use metrics::{
    MetricsEndpoint, MetricsPublishError, decode_metrics, encode_metrics, metrics_text,
};
pub use metrics_encoder::{MetricsEncodeError, MetricsEncoder};
pub use protocol::{
    ConversationRequest, FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob,
    PromptProgress, TokenLogprob,
};

pub const EVENT_QUEUE_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct RouteConfig {
    pub enable_openai: bool,
    pub enable_anthropic: bool,
    pub enable_demo: bool,
    pub routed_dp_size: usize,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            enable_openai: true,
            enable_anthropic: true,
            enable_demo: true,
            routed_dp_size: 1,
        }
    }
}

/// What `GET /v1/models` tells about the served model in addition to its id. The response
/// does not include a field that is `None`. `Some(false)` tells that the model does not
/// support the capability.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    /// The maximum number of prompt and generated tokens in one request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<NonZeroU32>,
    /// The maximum number of generated tokens in one request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<NonZeroU32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_vision: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_function_calling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning: Option<bool>,
    /// The `reasoning_effort` values that the model takes, from the least to the most effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_levels: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_prompt_caching: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<String>>,
    /// The sampling settings of a request that gives none: the model's recommended values
    /// (its `generation_config.json`). Without them, temperature 1.0, top_p 1.0 and no top_k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_top_k: Option<NonZeroU32>,
}

impl ModelMetadata {
    /// A request's temperature: the one it gives, else the model's default, else 1.0.
    pub fn temperature(&self, requested: Option<f64>) -> f64 {
        requested.or(self.default_temperature).unwrap_or(1.0)
    }

    /// A request's top_p: the one it gives, else the model's default, else 1.0.
    pub fn top_p(&self, requested: Option<f64>) -> f64 {
        requested.or(self.default_top_p).unwrap_or(1.0)
    }

    /// A request's top_k: the one it gives, else the model's default, else none.
    pub fn top_k(&self, requested: Option<u32>) -> Option<u32> {
        requested.or(self.default_top_k.map(NonZeroU32::get))
    }
}

/// The model that the server serves: the id that requests must name, and its metadata.
#[derive(Clone, Debug)]
pub struct ServedModel {
    pub id: Arc<str>,
    pub metadata: ModelMetadata,
}

impl ServedModel {
    pub fn new(id: impl Into<Arc<str>>, metadata: ModelMetadata) -> Self {
        Self {
            id: id.into(),
            metadata,
        }
    }
}

impl From<&str> for ServedModel {
    fn from(id: &str) -> Self {
        Self::new(id, ModelMetadata::default())
    }
}

impl From<String> for ServedModel {
    fn from(id: String) -> Self {
        Self::new(id, ModelMetadata::default())
    }
}

impl From<Arc<str>> for ServedModel {
    fn from(id: Arc<str>) -> Self {
        Self::new(id, ModelMetadata::default())
    }
}

/// One model of the server: what `GET /v1/models` tells about it, and the queue that its
/// requests go to.
pub struct ModelRoute {
    pub model: ServedModel,
    pub jobs: mpsc::Sender<GenerationJob>,
}

/// One served model's state in the handlers.
#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<GenerationJob>,
    model: Arc<str>,
    metadata: Arc<ModelMetadata>,
    next_id: Arc<AtomicU64>,
    routed_dp_size: usize,
}

/// Every served model, in registration order. A request goes to the model that it names.
#[derive(Clone)]
struct Models {
    routes: Arc<[AppState]>,
}

impl Models {
    fn first(&self) -> &AppState {
        &self.routes[0]
    }

    /// The state of the model that a request names, or HTTP 404 with the served ids.
    fn route(
        &self,
        model: &str,
        error_type: &str,
    ) -> Result<&AppState, Box<axum::response::Response>> {
        if let Some(state) = self
            .routes
            .iter()
            .find(|state| state.model.as_ref() == model)
        {
            return Ok(state);
        }
        let served = self
            .routes
            .iter()
            .map(|state| format!("{:?}", state.model))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Box::new(axum::response::IntoResponse::into_response((
            StatusCode::NOT_FOUND,
            axum::Json(protocol::json_error(
                format!("model {model:?} is not served; the server serves {served}"),
                error_type,
            )),
        ))))
    }
}

impl AppState {
    fn request_id(&self, prefix: &str) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{sequence:016x}")
    }

    fn enqueue(
        &self,
        request: ConversationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, (StatusCode, &'static str)> {
        let (events, receiver) = mpsc::channel(EVENT_QUEUE_CAPACITY);
        match self.jobs.try_send(GenerationJob {
            request,
            events,
            enqueued_at: Instant::now(),
        }) {
            Ok(()) => Ok(receiver),
            Err(mpsc::error::TrySendError::Full(_)) => Err((
                StatusCode::TOO_MANY_REQUESTS,
                "the inference request queue is full; retry later",
            )),
            Err(mpsc::error::TrySendError::Closed(_)) => Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "the inference engine is not running",
            )),
        }
    }
}

pub fn router(
    model: impl Into<ServedModel>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> Router {
    router_for(
        vec![ModelRoute {
            model: model.into(),
            jobs,
        }],
        routes,
        metrics,
    )
}

/// A router for several models, each with its own queue. Panics without a model or with
/// two models of one id.
pub fn router_for(
    models: Vec<ModelRoute>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> Router {
    assert!(!models.is_empty(), "a server serves at least one model");
    let next_id = Arc::new(AtomicU64::new(1));
    let states = models
        .into_iter()
        .map(|route| AppState {
            jobs: route.jobs,
            model: route.model.id,
            metadata: Arc::new(route.model.metadata),
            next_id: next_id.clone(),
            routed_dp_size: routes.routed_dp_size,
        })
        .collect::<Vec<_>>();
    for (index, state) in states.iter().enumerate() {
        assert!(
            states[..index]
                .iter()
                .all(|other| other.model != state.model),
            "model {:?} is served twice",
            state.model
        );
    }
    let state = Models {
        routes: states.into(),
    };
    let mut router = Router::new().route("/health", get(openai::health));
    if let Some(metrics) = metrics {
        router = router.route(
            "/metrics",
            get(move |headers: HeaderMap| {
                let metrics = metrics.clone();
                async move { metrics.response(&headers) }
            }),
        );
    }
    if routes.enable_demo {
        router = router.route("/", get(openai::index));
    }
    if routes.enable_openai {
        router = router
            .route("/v1/models", get(openai::models))
            .route("/v1/chat/completions", post(openai::chat_completions))
            .route("/tokenize", post(openai::tokenize))
            .route("/v1/responses", post(openai::responses));
    }
    if routes.enable_anthropic {
        router = router.route("/v1/messages", post(anthropic::messages));
    }
    router
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    model: impl Into<ServedModel>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> std::io::Result<()> {
    serve_with_shutdown(listener, model, jobs, routes, metrics, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

/// Stop accepting connections when requested and bound the time spent draining
/// existing HTTP connections. The hosting runtime may then be dropped even if
/// a client has stopped reading its response.
pub async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    model: impl Into<ServedModel>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    serve_with_drain(listener, model, jobs, routes, metrics, async {
        shutdown.await;
        Duration::from_secs(5)
    })
    .await
}

/// As [`serve_with_shutdown`], where the shutdown future also gives the drain time. A
/// long drain lets requests in flight finish.
pub async fn serve_with_drain(
    listener: tokio::net::TcpListener,
    model: impl Into<ServedModel>,
    jobs: mpsc::Sender<GenerationJob>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
    shutdown: impl Future<Output = Duration>,
) -> std::io::Result<()> {
    serve_models_with_drain(
        listener,
        vec![ModelRoute {
            model: model.into(),
            jobs,
        }],
        routes,
        metrics,
        shutdown,
    )
    .await
}

/// As [`serve_with_drain`] for several models: a request goes to the model that it names.
pub async fn serve_models_with_drain(
    listener: tokio::net::TcpListener,
    models: Vec<ModelRoute>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
    shutdown: impl Future<Output = Duration>,
) -> std::io::Result<()> {
    let (stop, stopped) = oneshot::channel::<()>();
    let server = axum::serve(listener, router_for(models, routes, metrics))
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        drain = shutdown => {
            let _ = stop.send(());
            tokio::time::timeout(drain, &mut server)
                .await
                .unwrap_or(Ok(()))
        }
    }
}
