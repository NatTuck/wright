use std::{
    net::{IpAddr, SocketAddr},
    num::NonZeroU32,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pythonize::depythonize;
use resin_core::model_share::ModelShare;
use resin_http::{
    GenerationJob, MetricsEndpoint, ModelMetadata, ModelRoute, RouteConfig, ServedModel,
};
use serde::Deserialize;
use tokio::sync::{mpsc, watch};

use crate::Engine;
use crate::http_session::Session;

pub(crate) const MAX_ACTIVE_REQUESTS: usize = 128;
pub(crate) const ADMISSIONS_PER_STEP: usize = 4;

/// One served model: its Engine, recipe and API facts, and its weight in the card's time.
struct ServedEngine {
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    model: String,
    metadata: ModelMetadata,
    stream_interval: u32,
    share: f64,
}

impl ServedEngine {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            engine: self.engine.clone_ref(py),
            recipe: self.recipe.clone_ref(py),
            model: self.model.clone(),
            metadata: self.metadata.clone(),
            stream_interval: self.stream_interval,
            share: self.share,
        }
    }
}

/// A model whose Engines run in other processes, one per card (`http_replicas`).
#[cfg(target_os = "linux")]
struct ReplicatedModel {
    model: String,
    router: Py<crate::http_replicas::ReplicaRouter>,
}

/// What `run_server` gets of a replicated model.
#[cfg(target_os = "linux")]
struct Replicated {
    model: String,
    metadata: ModelMetadata,
    replicas: Vec<crate::http_replicas::Replica>,
    slack: usize,
}
#[cfg(not(target_os = "linux"))]
struct Replicated {
    model: String,
    metadata: ModelMetadata,
}

#[pyclass(name = "_HttpServer", module = "resin._resin")]
pub(crate) struct NativeHttpServer {
    /// The first model, then those that `register` adds.
    models: Vec<ServedEngine>,
    distributed: Option<Py<PyAny>>,
    #[cfg(target_os = "linux")]
    replicated: Vec<ReplicatedModel>,
    request_queue_capacity: u32,
    drain_seconds: f64,
    routes: RouteConfig,
    running: AtomicBool,
    shutdown: Mutex<Option<watch::Sender<bool>>>,
}

/// The capabilities that the caller tells and the engine cannot: the keyword arguments of
/// resin.server.ModelCapabilities.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Capabilities {
    function_calling: Option<bool>,
    reasoning: Option<bool>,
    thinking_levels: Option<Vec<String>>,
    input_modalities: Option<Vec<String>>,
    output_modalities: Option<Vec<String>>,
    default_temperature: Option<f64>,
    default_top_p: Option<f64>,
    default_top_k: Option<u32>,
}

impl Capabilities {
    pub(crate) fn from_python(capabilities: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let Some(capabilities) = capabilities else {
            return Ok(Self::default());
        };
        let capabilities: Self = depythonize(capabilities)
            .map_err(|error| PyValueError::new_err(format!("invalid capabilities: {error}")))?;
        for names in [
            &capabilities.thinking_levels,
            &capabilities.input_modalities,
            &capabilities.output_modalities,
        ]
        .into_iter()
        .flatten()
        {
            if names.is_empty() || names.iter().any(|name| name.trim().is_empty()) {
                return Err(PyValueError::new_err(
                    "capabilities levels and modalities must be nonempty lists of nonempty names",
                ));
            }
        }
        if capabilities.thinking_levels.is_some() && capabilities.reasoning != Some(true) {
            return Err(PyValueError::new_err(
                "capabilities thinking_levels require reasoning=True",
            ));
        }
        if capabilities
            .default_temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(PyValueError::new_err(
                "capabilities default_temperature must be finite and non-negative",
            ));
        }
        if capabilities
            .default_top_p
            .is_some_and(|value| !value.is_finite() || !(0.0 < value && value <= 1.0))
        {
            return Err(PyValueError::new_err(
                "capabilities default_top_p must be finite and in (0, 1]",
            ));
        }
        if capabilities.default_top_k == Some(0) {
            return Err(PyValueError::new_err(
                "capabilities default_top_k must be positive",
            ));
        }
        Ok(capabilities)
    }
}

/// The metadata of `GET /v1/models`. The engine gives the token limits and prompt caching. A
/// request can generate tokens until its prompt and output fill the context, so the output
/// limit is the context length.
pub(crate) fn model_metadata(engine: &Engine, capabilities: Capabilities) -> ModelMetadata {
    let context = NonZeroU32::new(engine.context_length());
    ModelMetadata {
        max_context_tokens: context,
        max_output_tokens: context,
        supports_vision: capabilities
            .input_modalities
            .as_ref()
            .map(|modalities| modalities.iter().any(|name| name == "image")),
        supports_function_calling: capabilities.function_calling,
        supports_reasoning: capabilities.reasoning,
        thinking_levels: capabilities.thinking_levels,
        supports_prompt_caching: Some(engine.prefix_cache_stats().is_some()),
        input_modalities: capabilities.input_modalities,
        output_modalities: capabilities.output_modalities,
        default_temperature: capabilities.default_temperature,
        default_top_p: capabilities.default_top_p,
        default_top_k: capabilities.default_top_k.and_then(NonZeroU32::new),
    }
}

#[pymethods]
impl NativeHttpServer {
    #[new]
    #[pyo3(signature = (
        engine,
        recipe: "Recipe",
        *,
        model,
        stream_interval = 1,
        request_queue_capacity = 1024,
        drain_seconds = 30.0,
        enable_openai = true,
        enable_anthropic = true,
        enable_demo = true,
        distributed = None,
        capabilities = None,
        share = 1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        model: String,
        stream_interval: u32,
        request_queue_capacity: u32,
        drain_seconds: f64,
        enable_openai: bool,
        enable_anthropic: bool,
        enable_demo: bool,
        distributed: Option<Py<PyAny>>,
        capabilities: Option<Bound<'_, PyDict>>,
        share: f64,
    ) -> PyResult<Self> {
        if stream_interval == 0 {
            return Err(PyValueError::new_err("stream_interval must be positive"));
        }
        if request_queue_capacity == 0 {
            return Err(PyValueError::new_err(
                "request_queue_capacity must be positive",
            ));
        }
        if !drain_seconds.is_finite() || drain_seconds < 0.0 {
            return Err(PyValueError::new_err("drain_seconds must be zero or more"));
        }
        let routed_dp_size = distributed
            .as_ref()
            .map(|coordinator| {
                Python::attach(|py| coordinator.bind(py).getattr("size")?.extract::<usize>())
            })
            .transpose()?
            .unwrap_or(1);
        if routed_dp_size == 0 {
            return Err(PyValueError::new_err("routed_dp_size must be positive"));
        }
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be nonempty"));
        }
        if enable_demo && !enable_openai {
            return Err(PyValueError::new_err(
                "enable_demo=True requires enable_openai=True",
            ));
        }
        check_share(share)?;
        let capabilities = Capabilities::from_python(capabilities.as_ref())?;
        let metadata = model_metadata(&engine.bind(py).borrow(), capabilities);
        Ok(Self {
            models: vec![ServedEngine {
                engine,
                recipe,
                model,
                metadata,
                stream_interval,
                share,
            }],
            distributed,
            #[cfg(target_os = "linux")]
            replicated: Vec::new(),
            request_queue_capacity,
            drain_seconds,
            routes: RouteConfig {
                enable_openai,
                enable_anthropic,
                enable_demo,
                routed_dp_size,
            },
            running: AtomicBool::new(false),
            shutdown: Mutex::new(None),
        })
    }

    /// Serve one more model on its own Engine (before `serve`). `share` weighs its card time
    /// against the other models': the Engines step in turn on one card.
    #[pyo3(signature = (engine, recipe: "Recipe", *, model, stream_interval = 1, capabilities = None, share = 1.0))]
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        model: String,
        stream_interval: u32,
        capabilities: Option<Bound<'_, PyDict>>,
        share: f64,
    ) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err(
                "register models before the server runs",
            ));
        }
        if self.distributed.is_some() {
            return Err(PyValueError::new_err(
                "a distributed server serves one model",
            ));
        }
        if stream_interval == 0 {
            return Err(PyValueError::new_err("stream_interval must be positive"));
        }
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be nonempty"));
        }
        if self.models().contains(&model) {
            return Err(PyValueError::new_err(format!(
                "model {model:?} is already served"
            )));
        }
        if self.models.iter().any(|served| served.engine.is(&engine)) {
            return Err(PyValueError::new_err("each model needs its own Engine"));
        }
        check_share(share)?;
        let capabilities = Capabilities::from_python(capabilities.as_ref())?;
        let metadata = model_metadata(&engine.bind(py).borrow(), capabilities);
        self.models.push(ServedEngine {
            engine,
            recipe,
            model,
            metadata,
            stream_interval,
            share,
        });
        Ok(())
    }

    /// Serve one more model whose Engines run in other processes, one per card, through
    /// `router` (a `_ReplicaRouter` whose replicas have joined). It can sit beside a
    /// distributed model. Each request goes to one replica.
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (router, *, model))]
    fn register_replicas(
        &mut self,
        py: Python<'_>,
        router: Py<crate::http_replicas::ReplicaRouter>,
        model: String,
    ) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err(
                "register models before the server runs",
            ));
        }
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be nonempty"));
        }
        if self.models().contains(&model) {
            return Err(PyValueError::new_err(format!(
                "model {model:?} is already served"
            )));
        }
        if router.bind(py).borrow().metadata().is_none() {
            return Err(PyValueError::new_err(
                "accept the replicas before registering them",
            ));
        }
        self.replicated.push(ReplicatedModel { model, router });
        Ok(())
    }

    /// Serve until shutdown. `on_listening`, if given, is called (with no arguments) once the
    /// listener is bound and SIGTERM drains the server: from then on a client connects.
    #[pyo3(signature = (*, host = "127.0.0.1", port = 8000, on_listening = None))]
    fn serve(
        &self,
        py: Python<'_>,
        host: &str,
        port: u16,
        on_listening: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let host = host.parse::<IpAddr>().map_err(|_| {
            PyValueError::new_err("host must be an IPv4 or IPv6 address such as '0.0.0.0'")
        })?;
        let (shutdown, _) = watch::channel(false);
        {
            let mut slot = self
                .shutdown
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if self.running.swap(true, Ordering::AcqRel) {
                return Err(PyRuntimeError::new_err("HttpServer is already running"));
            }
            *slot = Some(shutdown.clone());
        }
        let models = self
            .models
            .iter()
            .map(|served| served.clone_ref(py))
            .collect::<Vec<_>>();
        let distributed = self.distributed.as_ref().map(|value| value.clone_ref(py));
        #[cfg(target_os = "linux")]
        let replicated = self
            .replicated
            .iter()
            .map(|served| {
                let (replicas, metadata, slack) = served.router.bind(py).borrow_mut().take()?;
                Ok(Replicated {
                    model: served.model.clone(),
                    metadata,
                    replicas,
                    slack,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        #[cfg(not(target_os = "linux"))]
        let replicated = Vec::new();
        let routes = self.routes;
        let request_queue_capacity = self.request_queue_capacity;
        let drain = Duration::from_secs_f64(self.drain_seconds);
        let result = py.detach(move || {
            run_server(
                models,
                distributed,
                replicated,
                routes,
                request_queue_capacity,
                drain,
                SocketAddr::new(host, port),
                shutdown,
                on_listening,
            )
        });
        {
            let mut slot = self
                .shutdown
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *slot = None;
            self.running.store(false, Ordering::Release);
        }
        result.map_err(|error| match error {
            ServeError::Message(message) => PyRuntimeError::new_err(message),
            ServeError::Python(error) => error,
        })
    }

    /// Request shutdown from another thread; serve returns after the current
    /// inference step has finished and its requests have been cancelled.
    fn shutdown(&self) {
        let slot = self
            .shutdown
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(shutdown) = slot.as_ref() {
            shutdown.send_replace(true);
        }
    }

    /// The first model's id.
    #[getter]
    fn model(&self) -> &str {
        &self.models[0].model
    }

    /// Every served model's id, in registration order (replicated models last).
    #[getter]
    fn models(&self) -> Vec<String> {
        let models = self.models.iter().map(|served| served.model.clone());
        #[cfg(target_os = "linux")]
        let models = models.chain(self.replicated.iter().map(|served| served.model.clone()));
        models.collect()
    }

    #[getter]
    fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn __repr__(&self) -> String {
        format!(
            "_HttpServer(models={:?}, stream_interval={}, request_queue_capacity={}, enable_openai={}, enable_anthropic={}, enable_demo={}, running={})",
            self.models().join(", "),
            self.models[0].stream_interval,
            self.request_queue_capacity,
            self.routes.enable_openai,
            self.routes.enable_anthropic,
            self.routes.enable_demo,
            self.running.load(Ordering::Acquire),
        )
    }
}

fn check_share(share: f64) -> PyResult<()> {
    if share.is_finite() && share > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "share must be a finite positive number",
        ))
    }
}

/// Why `run_server` failed: its own error, or the error of the `on_listening` callback.
enum ServeError {
    Message(String),
    Python(PyErr),
}

impl From<String> for ServeError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_server(
    models: Vec<ServedEngine>,
    distributed: Option<Py<PyAny>>,
    replicated: Vec<Replicated>,
    routes: RouteConfig,
    request_queue_capacity: u32,
    drain: Duration,
    address: SocketAddr,
    shutdown: watch::Sender<bool>,
    on_listening: Option<Py<PyAny>>,
) -> Result<(), ServeError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to create the HTTP runtime: {error}"))?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(address))
        .map_err(|error| format!("failed to bind {address}: {error}"))?;
    // One queue a model: a model with a full batch does not hold up the others' admissions.
    let (senders, mut receivers): (Vec<_>, Vec<_>) = (0..models.len() + replicated.len())
        .map(|_| mpsc::channel(request_queue_capacity as usize))
        .unzip();
    let replica_receivers = receivers.split_off(models.len());
    let metrics_enabled = Python::attach(|py| {
        models
            .iter()
            .any(|served| served.engine.bind(py).borrow().metrics_enabled())
    });
    let (endpoints, replica_endpoints): (Vec<Option<MetricsEndpoint>>, Vec<_>) = if metrics_enabled
    {
        let first = MetricsEndpoint::new(models[0].model.as_str());
        let mut endpoints = vec![Some(first.clone())];
        endpoints.extend(
            models[1..]
                .iter()
                .map(|served| Some(first.for_model(served.model.as_str()))),
        );
        let replicas = replicated
            .iter()
            .map(|served| Some(first.for_model(served.model.as_str())))
            .collect();
        (endpoints, replicas)
    } else {
        (vec![None; models.len()], vec![None; replicated.len()])
    };
    let names = models
        .iter()
        .map(|served| (served.model.as_str(), served.metadata.clone()))
        .chain(
            replicated
                .iter()
                .map(|served| (served.model.as_str(), served.metadata.clone())),
        )
        .map(|(model, metadata)| ServedModel::new(model, metadata))
        .collect::<Vec<_>>();
    let routes_for = names
        .into_iter()
        .zip(senders)
        .map(|(model, jobs)| ModelRoute { model, jobs })
        .collect::<Vec<_>>();
    let metrics = endpoints[0].clone();
    let actor_shutdown = shutdown.subscribe();
    // Any actor that ends stops the server.
    let (actor_finished, mut actor_done) = mpsc::channel::<()>(1 + replicated.len());
    let replica_actors = spawn_replica_actors(
        replicated,
        replica_receivers,
        replica_endpoints,
        &shutdown,
        &actor_finished,
    )?;
    let actor = thread::Builder::new()
        .name("resin-http-engine".to_owned())
        .spawn(move || {
            let result = if let Some(coordinator) = distributed {
                let receiver = receivers
                    .into_iter()
                    .next()
                    .expect("a distributed server serves one model");
                crate::http_distributed::run(
                    coordinator,
                    receiver,
                    actor_shutdown,
                    routes.routed_dp_size,
                    endpoints.into_iter().next().flatten(),
                )
            } else {
                engines_actor(models, receivers, actor_shutdown, endpoints)
            };
            if let Err(error) = &result {
                eprintln!("[resin-http] inference actor failed: {error}");
            }
            let _ = actor_finished.try_send(());
            result
        })
        .map_err(|error| format!("failed to start the inference thread: {error}"))?;
    // SIGTERM drains the server from here on, before a client can know that it listens.
    let terminate = {
        let _context = runtime.enter();
        Terminate::new()
    };
    // The listener is bound: the kernel takes connections, and the server accepts them below.
    let mut listening_error = None;
    if let Some(callback) = on_listening
        && let Err(error) = Python::attach(|py| callback.call0(py).map(drop))
    {
        shutdown.send_replace(true);
        listening_error = Some(error);
    }
    let mut stopping = shutdown.subscribe();
    let stop_actor = shutdown.clone();
    let server_result = runtime.block_on(resin_http::serve_models_with_drain(
        listener,
        routes_for,
        routes,
        metrics,
        async move {
            if !*stopping.borrow() {
                tokio::select! {
                    _ = actor_done.recv() => {},
                    _ = stopping.changed() => {},
                    _ = tokio::signal::ctrl_c() => {},
                    // SIGTERM stops new connections, and requests in flight run for the drain
                    // time. The inference actor keeps going until the server returns.
                    () = terminate.recv() => return drain,
                }
            }
            stop_actor.send_replace(true);
            Duration::from_secs(5)
        },
    ));
    shutdown.send_replace(true);
    let actor_result = actor
        .join()
        .map_err(|_| "the inference thread panicked".to_owned())?;
    let mut replica_result = Ok(());
    for actor in replica_actors {
        let result = actor
            .join()
            .map_err(|_| "a replica router thread panicked".to_owned())
            .and_then(|result| result);
        replica_result = replica_result.and(result);
    }
    if let Some(error) = listening_error {
        return Err(ServeError::Python(error));
    }
    server_result
        .map_err(|error| format!("HTTP server failed: {error}"))
        .and(actor_result)
        .and(replica_result)
        .map_err(ServeError::Message)
}

/// One router thread per replicated model (`http_replicas::run`).
fn spawn_replica_actors(
    replicated: Vec<Replicated>,
    receivers: Vec<mpsc::Receiver<GenerationJob>>,
    endpoints: Vec<Option<MetricsEndpoint>>,
    shutdown: &watch::Sender<bool>,
    finished: &mpsc::Sender<()>,
) -> Result<Vec<thread::JoinHandle<Result<(), String>>>, String> {
    let mut actors = Vec::new();
    for ((served, jobs), metrics) in replicated.into_iter().zip(receivers).zip(endpoints) {
        let shutdown = shutdown.subscribe();
        let finished = finished.clone();
        let actor = thread::Builder::new()
            .name("resin-http-replicas".to_owned())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                let result = crate::http_replicas::run(
                    served.replicas,
                    served.slack,
                    jobs,
                    shutdown,
                    metrics,
                )
                .map_err(|error| format!("{}: {error}", served.model));
                #[cfg(not(target_os = "linux"))]
                let result: Result<(), String> = {
                    let _ = (jobs, shutdown, metrics, served.metadata);
                    Err(format!("{}: replicas need Linux", served.model))
                };
                if let Err(error) = &result {
                    eprintln!("[resin-http] replica router failed: {error}");
                }
                let _ = finished.try_send(());
                result
            })
            .map_err(|error| format!("failed to start a replica router: {error}"))?;
        actors.push(actor);
    }
    Ok(actors)
}

/// SIGTERM, registered when made (inside the runtime) so that none is missed before the
/// server waits for it.
struct Terminate {
    #[cfg(unix)]
    signal: Option<tokio::signal::unix::Signal>,
}

impl Terminate {
    fn new() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            Self {
                signal: signal(SignalKind::terminate()).ok(),
            }
        }
        #[cfg(not(unix))]
        Self {}
    }

    async fn recv(mut self) {
        #[cfg(unix)]
        if let Some(signal) = self.signal.as_mut() {
            signal.recv().await;
            return;
        }
        std::future::pending().await
    }
}

/// One model in the inference actor.
struct Driven {
    engine: Py<Engine>,
    model: String,
    session: Session,
    jobs: mpsc::Receiver<GenerationJob>,
    metrics: Option<MetricsEndpoint>,
    disconnected: bool,
    /// The last step gave no events (a step that waits for cache, or the first step of an
    /// overlapped pair): another model steps before this one does again.
    deferred: bool,
}

/// Admit, step and stream the requests of every served model on one thread. The Engines share
/// the card, so they step one at a time; `ModelShare` picks the next one by card time.
fn engines_actor(
    models: Vec<ServedEngine>,
    receivers: Vec<mpsc::Receiver<GenerationJob>>,
    shutdown: watch::Receiver<bool>,
    endpoints: Vec<Option<MetricsEndpoint>>,
) -> Result<(), String> {
    let shares = models.iter().map(|served| served.share).collect::<Vec<_>>();
    let mut share = ModelShare::new(&shares).map_err(|error| error.to_string())?;
    let mut driven = Vec::with_capacity(models.len());
    for ((served, jobs), metrics) in models.into_iter().zip(receivers).zip(endpoints) {
        let mut session = Python::attach(|py| {
            let mut session = Session::new(
                py,
                served.engine.clone_ref(py),
                served.recipe,
                served.stream_interval,
            )?;
            // a prompt renders beside the loop, not between two steps
            session.render_aside(py)?;
            Ok::<_, PyErr>(session)
        })
        .map_err(|error| format_python_error("HTTP session preparation failed", error))?;
        session.metrics = metrics.clone();
        driven.push(Driven {
            engine: served.engine,
            model: served.model,
            session,
            jobs,
            metrics,
            disconnected: false,
            deferred: false,
        });
    }
    let mut next_metrics = Instant::now();
    loop {
        if driven.iter().any(|model| model.metrics.is_some()) {
            for model in &driven {
                if let Some(endpoint) = &model.metrics {
                    endpoint.set_queue_size(model.jobs.len());
                }
            }
            if Instant::now() >= next_metrics {
                let mut snapshots = Vec::with_capacity(driven.len());
                for model in &driven {
                    let snapshot = Python::attach(|py| {
                        let engine = model.engine.bind(py).borrow();
                        match engine.collect_metrics(py, true)? {
                            Some(metrics) => metrics.encode_protobuf(&model.model, 0, 0).map(Some),
                            None => Ok(None),
                        }
                    })
                    .map_err(|error| format_python_error("metrics collection failed", error))?;
                    snapshots.extend(snapshot);
                }
                if let Some(endpoint) = driven.iter().find_map(|model| model.metrics.as_ref()) {
                    endpoint
                        .publish(snapshots)
                        .map_err(|error| error.to_string())?;
                }
                next_metrics = Instant::now() + Duration::from_secs(1);
            }
        }
        let stopping = *shutdown.borrow();
        for model in &mut driven {
            for _ in 0..ADMISSIONS_PER_STEP {
                // (at most ADMISSIONS_PER_STEP jobs render at a time; the others wait in the queue, which
                // turns requests away when it is full)
                if model.disconnected
                    || stopping
                    || model.session.active_count() >= MAX_ACTIVE_REQUESTS
                    || model.session.rendering() >= ADMISSIONS_PER_STEP
                {
                    break;
                }
                match model.jobs.try_recv() {
                    Ok(job) => model.session.admit(job),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        model.disconnected = true;
                        break;
                    }
                }
            }
        }
        if stopping || driven.iter().any(|model| model.disconnected) {
            for model in &mut driven {
                model.session.close().map_err(|error| {
                    format_python_error("failed to cancel HTTP requests", error)
                })?;
            }
            for model in &driven {
                Python::attach(|py| model.engine.bind(py).borrow_mut().close(py)).map_err(
                    |error| format_python_error("failed to stop the inference engine", error),
                )?;
            }
            return Ok(());
        }
        let mut ready = Vec::with_capacity(driven.len());
        for model in &mut driven {
            model.session.poll_rendered();
            model.session.cancel_disconnected()?;
            // (a model whose requests still render waits: the loop sleeps and polls again)
            ready.push(model.session.submitted_count() > 0 || engine_has_work(&model.engine)?);
        }
        let others = ready.iter().filter(|&&ready| ready).count() > 1;
        let pick = ready
            .iter()
            .zip(&driven)
            .map(|(&ready, model)| ready && !(others && model.deferred))
            .collect::<Vec<_>>();
        let Some(index) = share.next(&pick) else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };
        let started = Instant::now();
        let token_events = Python::attach(|py| {
            let mut engine = driven[index].engine.bind(py).borrow_mut();
            engine.step(py, None, true).map(|events| {
                events
                    .into_iter()
                    .map(|event| {
                        let event = event.bind(py).borrow();
                        event.clone()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .map_err(|error| {
            format_python_error(
                &format!("engine step of {} failed", driven[index].model),
                error,
            )
        })?;
        share.charge(index, started.elapsed().as_secs_f64());
        for (other, model) in driven.iter_mut().enumerate() {
            if other != index {
                model.deferred = false;
            }
        }
        driven[index].deferred = token_events.is_empty();
        if token_events.is_empty() && !others {
            thread::sleep(Duration::from_micros(50));
        }
        driven[index].session.process(token_events)?;
    }
}

fn engine_has_work(engine: &Py<Engine>) -> Result<bool, String> {
    Ok(Python::attach(|py| engine.bind(py).borrow().has_work()))
}

fn format_python_error(context: &str, error: PyErr) -> String {
    format!("{context}: {error}")
}
