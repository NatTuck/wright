//! Replicas: one model's one-card Engines, each in the process that holds its card.
//!
//! The HTTP frontend (rank 0) keeps the clients. A router thread sends each request to one
//! replica, which renders, schedules and decodes it with its own Engine and HTTP session, and
//! sends the request's events back. Nothing waits for the replicas to step together.
//!
//! Each replica connects to the router's socket (`step_channel::Listener`, the run token, the
//! same user) and sends READY with its model's metadata. Then the router sends INGRESS frames
//! (`HttpIngress` JSON: new requests, cancellations, shutdown), and the replica sends EVENTS
//! frames (`HttpEvents` JSON) after each step that has events. A replica whose Engine fails
//! sends FAILED and stops; the router ends its requests with the error and routes around it.
//!
//! The router never blocks on one replica: each replica has its own queue of frames to send,
//! which the router writes as that replica's socket takes them, and it reads every replica's
//! events in the meantime.
//!
//! A replica on a card that a tensor-parallel model also uses steps only on its turns: its gate
//! (`resin_models_gaudi.synapse.card_gate.CardGate`) says when. While it waits for a turn, it
//! still reads the router's frames, so that its socket never fills.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use resin_http::{
    ConversationRequest, GenerationEvent, GenerationJob, MetricsEndpoint, ModelMetadata,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::Engine;

/// A gate that a replica takes a turn at before each Engine step and gives back after it (a native card gate:
/// no Python in the replica's loop).
pub(crate) trait StepGate: Send + Sync {
    fn acquire(&self, timeout: Duration) -> Result<bool, String>;
    fn release(&self) -> Result<(), String>;
}
use crate::http::{ADMISSIONS_PER_STEP, Capabilities, MAX_ACTIVE_REQUESTS, model_metadata};
use crate::http_distributed::{HttpEvents, HttpIngress};
use crate::http_session::SessionCore;
use crate::step_channel::frame::{Frame, Header};
use crate::step_channel::socket::Peer;
use crate::step_channel::{Listener, connect, io_error, wait_frame};

const INGRESS: u8 = 2;
const READY: u8 = 3;
const EVENTS: u8 = 4;
const FAILED: u8 = 5;
const CLOSED: u8 = 6;

/// A peer that reads none of the frames sent to it for this long has failed.
const WRITE_TIMEOUT: Duration = Duration::from_secs(120);

/// How long a gated replica waits for its turn before it reads the router's frames again.
const GATE_WAIT: Duration = Duration::from_millis(20);

/// What a replica tells the router when it joins.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    metadata: ModelMetadata,
    max_running_requests: usize,
}

fn json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| error.to_string())
}

fn send(peer: &Peer, kind: u8, seq: u64, payload: &[u8]) -> std::io::Result<()> {
    let header = Header {
        kind,
        rank: 0,
        status: 0,
        seq,
    };
    peer.send(header, &[payload], Instant::now() + WRITE_TIMEOUT)
}

/// The replica end: serves one Engine's requests from the router until shutdown.
#[pyclass(name = "_ReplicaServer", module = "resin._resin")]
pub(crate) struct ReplicaServer {
    peer: Peer,
    index: u16,
    model: String,
    engine: Py<Engine>,
    core: Option<SessionCore>,
    seq: u64,
    gate: Option<Gate>,
}

/// A replica's gate: native (`resin._resin.CardGate`), or any Python object with `acquire` and `release`.
enum Gate {
    #[cfg_attr(not(feature = "gaudi"), allow(dead_code))]
    Native(Arc<dyn StepGate>),
    Python(Py<PyAny>),
}

impl Gate {
    fn of(gate: Py<PyAny>, py: Python<'_>) -> Self {
        #[cfg(feature = "gaudi")]
        if let Ok(native) = gate.bind(py).cast::<crate::gaudi_gate::CardGate>() {
            let inner: Arc<dyn StepGate> = native.get().inner.clone();
            return Self::Native(inner);
        }
        let _ = py;
        Self::Python(gate)
    }

    fn acquire(&self, py: Python<'_>, timeout: Duration) -> PyResult<bool> {
        match self {
            Self::Native(gate) => py
                .detach(|| gate.acquire(timeout))
                .map_err(PyRuntimeError::new_err),
            Self::Python(gate) => gate
                .bind(py)
                .call_method1("acquire", (timeout.as_secs_f64(),))?
                .is_truthy(),
        }
    }

    fn release(&self, py: Python<'_>) -> PyResult<()> {
        match self {
            Self::Native(gate) => py
                .detach(|| gate.release())
                .map_err(PyRuntimeError::new_err),
            Self::Python(gate) => gate.bind(py).call_method0("release").map(drop),
        }
    }
}

#[pymethods]
impl ReplicaServer {
    /// Join the router at `path` as replica `index` (from 0) and send READY. With a `gate`, the
    /// replica steps only when `gate.acquire(timeout)` returns true, and calls `gate.release()`
    /// after each step.
    #[new]
    #[pyo3(signature = (
        path: "str | os.PathLike[str]",
        token: "bytes",
        index: "int",
        engine,
        recipe: "Recipe",
        *,
        model = "resin".to_owned(),
        stream_interval = 1,
        capabilities = None,
        timeout = 600.0,
        gate = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        token: Vec<u8>,
        index: u16,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        model: String,
        stream_interval: u32,
        capabilities: Option<Bound<'_, PyDict>>,
        timeout: f64,
        gate: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let capabilities = Capabilities::from_python(capabilities.as_ref())?;
        let (ready, metrics_enabled) = {
            let engine = engine.bind(py).borrow();
            let ready = Ready {
                metadata: model_metadata(&engine, capabilities),
                max_running_requests: engine.max_running_requests(),
            };
            (ready, engine.metrics_enabled())
        };
        // With metrics, the session sends FirstToken, so that the router records the time to
        // the first token.
        let mut core = SessionCore::new(
            py,
            engine.clone_ref(py),
            recipe,
            stream_interval,
            metrics_enabled,
        )?;
        // a prompt renders beside the replica's loop, not between two of its steps
        core.render_aside(py)?;
        let rank = index
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("replica index out of range"))?;
        let peer = connect(py, &path, &token, rank, &[], timeout)?;
        let payload = json(&ready).map_err(PyRuntimeError::new_err)?;
        py.detach(|| send(&peer, READY, 0, &payload))
            .map_err(|error| io_error("sending READY to the router".into(), error))?;
        Ok(Self {
            peer,
            index,
            model,
            engine,
            core: Some(core),
            seq: 0,
            gate: gate.map(|gate| Gate::of(gate, py)),
        })
    }

    /// Serve until the router asks this replica to stop. An error stops the replica: it tells
    /// the router, cancels its requests and raises.
    #[pyo3(signature = () -> "None")]
    fn serve(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = self.run(py);
        if let Err(error) = &result {
            let message = format!("replica {}: {error}", self.index);
            self.seq += 1;
            let _ = py.detach(|| send(&self.peer, FAILED, self.seq, message.as_bytes()));
            if let Some(core) = &mut self.core {
                let _ = core.close();
            }
        }
        result
    }

    /// The router's index of this replica.
    #[getter]
    fn index(&self) -> u16 {
        self.index
    }
}

impl ReplicaServer {
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(core) = &mut self.core else {
            return Err(PyRuntimeError::new_err("the replica has stopped"));
        };
        loop {
            // Take every frame that has arrived. Wait for one only with nothing to step, nothing
            // just admitted (an admission can have events without a step) and nothing rendering.
            let mut admitted = false;
            let mut snapshot = false;
            loop {
                let busy =
                    admitted || core.rendering() > 0 || self.engine.bind(py).borrow().has_work();
                let frame = if busy || self.peer.buffered() {
                    self.peer
                        .try_frame()
                        .map_err(|error| io_error("reading from the router".into(), error))?
                } else {
                    Some(wait_frame(py, &mut self.peer, None, "requests")?)
                };
                let Some(frame) = frame else { break };
                if frame.header.kind != INGRESS {
                    return Err(PyRuntimeError::new_err(format!(
                        "replica protocol error: frame kind {} from the router",
                        frame.header.kind
                    )));
                }
                let ingress: HttpIngress = serde_json::from_slice(&frame.payload)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                if ingress.shutdown {
                    core.close()?;
                    self.engine.bind(py).borrow_mut().close(py)?;
                    self.seq += 1;
                    let seq = self.seq;
                    let peer = &self.peer;
                    py.detach(|| send(peer, CLOSED, seq, &[]))
                        .map_err(|error| io_error("sending CLOSED".into(), error))?;
                    return Ok(());
                }
                core.admit(&ingress, 0)?;
                admitted = true;
                snapshot |= ingress.collect_metrics;
            }
            core.poll_rendered();
            let has_work = self.engine.bind(py).borrow().has_work();
            // A gated replica that gets no turn reads the router's frames, then asks again.
            let turn = has_work
                && match &self.gate {
                    None => true,
                    Some(gate) => gate.acquire(py, GATE_WAIT)?,
                };
            let events = if turn {
                let stepped = self.engine.bind(py).borrow_mut().step(py, None, true);
                if let Some(gate) = &self.gate {
                    gate.release(py)?;
                }
                stepped?
                    .into_iter()
                    .map(|event| event.bind(py).borrow().clone())
                    .collect()
            } else {
                Vec::new()
            };
            let mut packet = core.process(events).map_err(PyRuntimeError::new_err)?;
            if snapshot {
                let engine = self.engine.bind(py).borrow();
                if let Some(metrics) = engine.collect_metrics(py, true)? {
                    packet.metrics.push(metrics.encode_protobuf(
                        &self.model,
                        0,
                        self.index as u32,
                    )?);
                }
            }
            if !packet.events.is_empty() || !packet.metrics.is_empty() {
                let payload = json(&packet).map_err(PyRuntimeError::new_err)?;
                self.seq += 1;
                let seq = self.seq;
                let peer = &self.peer;
                py.detach(|| send(peer, EVENTS, seq, &payload))
                    .map_err(|error| io_error("sending EVENTS to the router".into(), error))?;
            } else if turn {
                // A step without events (a cache wait, the first of an overlapped pair).
                py.detach(|| std::thread::sleep(Duration::from_micros(50)));
            } else if !has_work && core.rendering() > 0 {
                // nothing to step until a render is done
                py.detach(|| std::thread::sleep(Duration::from_micros(100)));
            }
        }
    }
}

/// The router end: accepts the replicas of one model, then belongs to the HTTP server.
#[pyclass(name = "_ReplicaRouter", module = "resin._resin")]
pub(crate) struct ReplicaRouter {
    listener: Listener,
    count: usize,
    joined: Option<(Vec<Replica>, Ready)>,
}

#[pymethods]
impl ReplicaRouter {
    /// Listen on `path` for `count` replicas; `token` is the secret they must send.
    #[new]
    #[pyo3(signature = (path: "str | os.PathLike[str]", token: "bytes", count: "int"))]
    fn new(path: PathBuf, token: Vec<u8>, count: u16) -> PyResult<Self> {
        if count == 0 {
            return Err(PyValueError::new_err("a replicated model needs a replica"));
        }
        Ok(Self {
            listener: Listener::new(path, token, (1..=count).collect())?,
            count: count as usize,
            joined: None,
        })
    }

    /// Wait until every replica has joined and sent READY; return the model's metadata as JSON.
    /// Every replica must report the same metadata.
    #[pyo3(signature = (timeout: "float") -> "str")]
    fn accept(&mut self, py: Python<'_>, timeout: f64) -> PyResult<String> {
        let until = Instant::now() + Duration::from_secs_f64(timeout.max(0.0));
        let mut replicas = Vec::with_capacity(self.count);
        let mut first: Option<Ready> = None;
        for (mut peer, _) in self.listener.accept(py, timeout)? {
            let frame = wait_frame(py, &mut peer, Some(until), "READY")?;
            if frame.header.kind == FAILED {
                return Err(PyRuntimeError::new_err(
                    String::from_utf8_lossy(&frame.payload).into_owned(),
                ));
            }
            if frame.header.kind != READY {
                return Err(PyRuntimeError::new_err(format!(
                    "replica protocol error: frame kind {} before READY",
                    frame.header.kind
                )));
            }
            let ready: Ready = serde_json::from_slice(&frame.payload)
                .map_err(|error| PyValueError::new_err(format!("replica READY: {error}")))?;
            if first.as_ref().is_some_and(|first| *first != ready) {
                return Err(PyValueError::new_err(format!(
                    "replica {} reports other settings than replica 0",
                    peer.rank - 1
                )));
            }
            first.get_or_insert(ready);
            replicas.push(Replica::new(peer));
        }
        let ready = first.expect("at least one replica");
        let metadata = serde_json::to_string(&ready.metadata)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        self.joined = Some((replicas, ready));
        Ok(metadata)
    }

    #[getter]
    fn count(&self) -> usize {
        self.count
    }
}

impl ReplicaRouter {
    /// The accepted replicas, their metadata and the load slack of affinity routing.
    pub(crate) fn take(&mut self) -> PyResult<(Vec<Replica>, ModelMetadata, usize)> {
        let (replicas, ready) = self.joined.take().ok_or_else(|| {
            PyRuntimeError::new_err("accept the replicas before serving, and serve them once")
        })?;
        let slack = (ready.max_running_requests / 8).max(2);
        Ok((replicas, ready.metadata, slack))
    }

    pub(crate) fn metadata(&self) -> Option<ModelMetadata> {
        self.joined
            .as_ref()
            .map(|(_, ready)| ready.metadata.clone())
    }
}

pub(crate) struct Replica {
    peer: Peer,
    seq: u64,
    /// Requests sent and not yet finished.
    outstanding: usize,
    state: ReplicaState,
    /// Frames to send: `outbox[written..]` is not written yet.
    outbox: Vec<u8>,
    written: usize,
    /// Since when the socket has taken none of the outbox.
    stalled: Option<Instant>,
    /// Whether the router has ended the requests of this failed replica.
    reported: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum ReplicaState {
    Serving,
    Closing,
    Closed,
    Failed(String),
}

impl Replica {
    fn new(peer: Peer) -> Self {
        Self {
            peer,
            seq: 0,
            outstanding: 0,
            state: ReplicaState::Serving,
            outbox: Vec::new(),
            written: 0,
            stalled: None,
            reported: false,
        }
    }

    fn index(&self) -> usize {
        self.peer.rank as usize - 1
    }

    fn active(&self) -> bool {
        matches!(self.state, ReplicaState::Serving | ReplicaState::Closing)
    }

    /// Add one frame to the outbox; `flush` sends it.
    fn queue(&mut self, kind: u8, payload: &[u8]) {
        self.seq += 1;
        let header = Header {
            kind,
            rank: 0,
            status: 0,
            seq: self.seq,
        };
        self.outbox.extend_from_slice(&header.lead(payload.len()));
        self.outbox.extend_from_slice(payload);
    }

    fn pending(&self) -> bool {
        self.written < self.outbox.len()
    }

    /// Write as much of the outbox as the socket takes now, without blocking. A replica that
    /// takes none of it for `WRITE_TIMEOUT` has failed.
    fn flush(&mut self) -> Result<(), String> {
        let fd = self.peer.stream.as_raw_fd();
        while self.pending() {
            let rest = &self.outbox[self.written..];
            // SAFETY: send reads at most rest.len() bytes from rest.
            let count = unsafe {
                libc::send(
                    fd,
                    rest.as_ptr() as *const libc::c_void,
                    rest.len(),
                    libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                )
            };
            if count < 0 {
                let error = std::io::Error::last_os_error();
                match error.kind() {
                    std::io::ErrorKind::Interrupted => continue,
                    std::io::ErrorKind::WouldBlock => break,
                    _ => {
                        return Err(format!(
                            "replica {}: sending requests failed: {error}",
                            self.index()
                        ));
                    }
                }
            }
            if count == 0 {
                break;
            }
            self.written += count as usize;
            self.stalled = None;
        }
        if !self.pending() {
            self.outbox.clear();
            self.written = 0;
            return Ok(());
        }
        if self.written >= 1 << 20 {
            self.outbox.drain(..self.written);
            self.written = 0;
        }
        let since = *self.stalled.get_or_insert_with(Instant::now);
        if since.elapsed() >= WRITE_TIMEOUT {
            return Err(format!(
                "replica {} read none of its requests for {} s",
                self.index(),
                WRITE_TIMEOUT.as_secs()
            ));
        }
        Ok(())
    }
}

/// Routes a conversation to the replica that holds its prefix cache: rendezvous hashing of its
/// key over the serving replicas, unless that replica is busier than the least busy one by
/// more than `slack` requests. A key moved for load stays on its new replica.
pub(crate) struct Affinity {
    slack: usize,
    moved: HashMap<u64, usize>,
}

/// The key of a request's conversation: a session id the client gives (`prompt_cache_key`,
/// `user`, or Anthropic `metadata.user_id`), else the tools and the messages up to the first
/// user message, which the later turns of a conversation repeat.
pub(crate) fn conversation_key(request: &ConversationRequest) -> u64 {
    let mut hasher = DefaultHasher::new();
    let session = ["prompt_cache_key", "user"]
        .iter()
        .find_map(|name| request.extra.get(*name).and_then(|value| value.as_str()))
        .or_else(|| {
            request
                .extra
                .get("metadata")
                .and_then(|metadata| metadata.get("user_id"))
                .and_then(|value| value.as_str())
        });
    match session {
        Some(session) => {
            0u8.hash(&mut hasher);
            session.hash(&mut hasher);
        }
        None => {
            1u8.hash(&mut hasher);
            for tool in &request.tools {
                tool.to_string().hash(&mut hasher);
            }
            let opening = request
                .messages
                .iter()
                .position(|message| {
                    message.get("role").and_then(|role| role.as_str()) == Some("user")
                })
                .map_or(request.messages.len(), |first_user| first_user + 1);
            for message in &request.messages[..opening] {
                message.to_string().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn rendezvous(key: u64, replica: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    (key, replica as u64).hash(&mut hasher);
    hasher.finish()
}

impl Affinity {
    pub(crate) fn new(slack: usize) -> Self {
        Self {
            slack,
            moved: HashMap::new(),
        }
    }

    /// The replica for a request with `key`, given each replica's outstanding requests, or None
    /// when no replica serves (`None` loads).
    pub(crate) fn pick(&mut self, key: u64, loads: &[Option<usize>]) -> Option<usize> {
        let serving = || {
            loads
                .iter()
                .enumerate()
                .filter_map(|(i, load)| load.map(|l| (i, l)))
        };
        let least = serving().min_by_key(|&(index, load)| (load, index))?;
        let preferred = self
            .moved
            .get(&key)
            .copied()
            .filter(|&index| loads.get(index).is_some_and(Option::is_some))
            .unwrap_or_else(|| {
                serving()
                    .max_by_key(|&(index, _)| rendezvous(key, index))
                    .expect("a serving replica")
                    .0
            });
        let load = loads[preferred].expect("a serving replica");
        if load <= least.1 + self.slack {
            return Some(preferred);
        }
        if self.moved.len() >= 1 << 16 {
            self.moved.clear();
        }
        self.moved.insert(key, least.0);
        Some(least.0)
    }
}

struct Client {
    replica: usize,
    events: mpsc::Sender<GenerationEvent>,
    started: Instant,
}

/// Route one model's HTTP jobs to its replicas until shutdown and every replica has closed.
pub(crate) fn run(
    mut replicas: Vec<Replica>,
    slack: usize,
    mut jobs: mpsc::Receiver<GenerationJob>,
    shutdown: watch::Receiver<bool>,
    metrics: Option<MetricsEndpoint>,
) -> Result<(), String> {
    let mut clients = HashMap::<String, Client>::new();
    let mut affinity = Affinity::new(slack);
    let mut stopping = false;
    // Requests whose client fell behind: cancelled on their replica in the next round.
    let mut cancel_next: Vec<(usize, String)> = Vec::new();
    // Each replica's last engine snapshot; the replicas send one when asked, once a second.
    let mut snapshots: Vec<Vec<Vec<u8>>> = replicas.iter().map(|_| Vec::new()).collect();
    let mut next_metrics = Instant::now();
    loop {
        if let Some(endpoint) = &metrics {
            endpoint.set_queue_size(jobs.len());
        }
        let collect_metrics = metrics.is_some() && Instant::now() >= next_metrics;
        if collect_metrics {
            next_metrics = Instant::now() + Duration::from_secs(1);
        }
        let mut ingress: Vec<HttpIngress> =
            replicas.iter().map(|_| HttpIngress::default()).collect();
        for (index, id) in cancel_next.drain(..) {
            ingress[index].canceled.push(id);
        }
        let closed: Vec<String> = clients
            .iter()
            .filter_map(|(id, client)| client.events.is_closed().then_some(id.clone()))
            .collect();
        for id in closed {
            let client = clients.remove(&id).expect("a client");
            let replica = &mut replicas[client.replica];
            replica.outstanding = replica.outstanding.saturating_sub(1);
            ingress[client.replica].canceled.push(id);
        }
        stopping |= *shutdown.borrow();
        let serving = replicas
            .iter()
            .filter(|replica| replica.state == ReplicaState::Serving)
            .count();
        // With no replica serving, every waiting job fails at once.
        for _ in 0..ADMISSIONS_PER_STEP.saturating_mul(serving.max(1)) {
            if stopping || (serving > 0 && clients.len() >= MAX_ACTIVE_REQUESTS * serving) {
                break;
            }
            let mut job = match jobs.try_recv() {
                Ok(job) => job,
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    stopping = true;
                    break;
                }
            };
            if job.events.is_closed() {
                continue;
            }
            if clients.contains_key(&job.request.id) {
                let _ = job.events.try_send(GenerationEvent::Error {
                    message: "HTTP request IDs must be unique strings".to_owned(),
                });
                continue;
            }
            let loads: Vec<Option<usize>> = replicas
                .iter()
                .map(|replica| {
                    (replica.state == ReplicaState::Serving).then_some(replica.outstanding)
                })
                .collect();
            let Some(index) = affinity.pick(conversation_key(&job.request), &loads) else {
                let _ = job.events.try_send(GenerationEvent::Error {
                    message: "no replica of this model is serving".to_owned(),
                });
                continue;
            };
            if let Some(endpoint) = &metrics {
                endpoint.observe_queue_time(job.enqueued_at.elapsed().as_secs_f64());
            }
            job.request.routed_dp_rank = None;
            replicas[index].outstanding += 1;
            clients.insert(
                job.request.id.clone(),
                Client {
                    replica: index,
                    events: job.events,
                    started: job.enqueued_at,
                },
            );
            ingress[index].requests.push(job.request);
        }
        for (index, replica) in replicas.iter_mut().enumerate() {
            let message = &mut ingress[index];
            message.collect_metrics = collect_metrics;
            if stopping && replica.state == ReplicaState::Serving {
                message.shutdown = true;
                replica.state = ReplicaState::Closing;
            } else if replica.state != ReplicaState::Serving {
                continue;
            }
            if message.requests.is_empty()
                && message.canceled.is_empty()
                && !message.shutdown
                && !message.collect_metrics
            {
                continue;
            }
            match json(&*message) {
                Ok(payload) => replica.queue(INGRESS, &payload),
                Err(error) => {
                    replica.state =
                        ReplicaState::Failed(format!("replica {index}: requests: {error}"))
                }
            }
        }
        for replica in &mut replicas {
            if replica.active()
                && replica.pending()
                && let Err(error) = replica.flush()
            {
                replica.state = ReplicaState::Failed(error);
            }
        }
        wait_readable(&replicas, clients.is_empty() && !stopping);
        for (index, replica) in replicas.iter_mut().enumerate() {
            loop {
                if !replica.active() {
                    break;
                }
                let frame = match replica.peer.try_frame() {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break,
                    Err(error) => {
                        replica.state = ReplicaState::Failed(format!(
                            "replica {}: the connection failed: {error}",
                            replica.index()
                        ));
                        break;
                    }
                };
                let update = receive(
                    replica,
                    frame,
                    &mut clients,
                    &mut cancel_next,
                    metrics.as_ref(),
                );
                if let Some(update) = update {
                    snapshots[index] = update;
                    if let Some(endpoint) = &metrics
                        && let Err(error) =
                            endpoint.publish(snapshots.iter().flatten().cloned().collect())
                    {
                        replica.state = ReplicaState::Failed(format!(
                            "replica {index}: its metrics are not valid: {error}"
                        ));
                    }
                }
            }
            if let ReplicaState::Failed(message) = &replica.state
                && !replica.reported
            {
                // Its requests end with the error, its metrics leave /metrics, and the others
                // keep serving. The error is logged here; the server still stops cleanly.
                replica.reported = true;
                let failed: Vec<String> = clients
                    .iter()
                    .filter(|(_, client)| client.replica == index)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in failed {
                    let client = clients.remove(&id).expect("a client");
                    let _ = client.events.try_send(GenerationEvent::Error {
                        message: message.clone(),
                    });
                }
                eprintln!("[resin-http] {message}");
                replica.outstanding = 0;
                snapshots[index].clear();
                if let Some(endpoint) = &metrics
                    && let Err(error) =
                        endpoint.publish(snapshots.iter().flatten().cloned().collect())
                {
                    eprintln!("[resin-http] replica metrics: {error}");
                }
            }
        }
        if stopping && replicas.iter().all(|replica| !replica.active()) {
            return Ok(());
        }
    }
}

/// Dispatch one frame from `replica`; return its metrics snapshot if it has one. A frame that
/// is not valid fails this replica only.
fn receive(
    replica: &mut Replica,
    frame: Frame,
    clients: &mut HashMap<String, Client>,
    cancel_next: &mut Vec<(usize, String)>,
    metrics: Option<&MetricsEndpoint>,
) -> Option<Vec<Vec<u8>>> {
    let mut snapshot = None;
    match frame.header.kind {
        EVENTS => {
            let packet: HttpEvents = match serde_json::from_slice(&frame.payload) {
                Ok(packet) => packet,
                Err(error) => {
                    replica.state = ReplicaState::Failed(format!(
                        "replica {}: events that are not valid: {error}",
                        replica.index()
                    ));
                    return None;
                }
            };
            if !packet.metrics.is_empty() {
                snapshot = Some(packet.metrics);
            }
            for (id, event) in packet.events {
                let terminal = matches!(
                    event,
                    GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
                );
                let mut behind = false;
                if let Some(client) = clients.get(&id) {
                    if let Some(endpoint) = metrics {
                        match &event {
                            GenerationEvent::FirstToken => {
                                endpoint.observe_first_token(client.started.elapsed().as_secs_f64())
                            }
                            GenerationEvent::Finished { .. } => {
                                endpoint.observe_completed(client.started.elapsed().as_secs_f64())
                            }
                            _ => {}
                        }
                    }
                    // Backpressure stays with one client: a client that falls behind is
                    // dropped, and its request is cancelled on the replica next round.
                    // A progress update is optional: a full queue drops only the update.
                    behind = match event {
                        GenerationEvent::FirstToken => false,
                        GenerationEvent::PromptProgress(_) => {
                            let _ = client.events.try_send(event);
                            false
                        }
                        event => client.events.try_send(event).is_err(),
                    };
                }
                if (terminal || behind) && clients.remove(&id).is_some() {
                    replica.outstanding = replica.outstanding.saturating_sub(1);
                    if !terminal {
                        cancel_next.push((replica.index(), id));
                    }
                }
            }
        }
        CLOSED => replica.state = ReplicaState::Closed,
        FAILED => {
            replica.state =
                ReplicaState::Failed(String::from_utf8_lossy(&frame.payload).into_owned())
        }
        kind => {
            replica.state = ReplicaState::Failed(format!(
                "replica {}: protocol error: frame kind {kind}",
                replica.index()
            ))
        }
    }
    snapshot
}

/// Wait up to a millisecond (longer when idle) for any replica to have a frame, or to take
/// more of its outbox.
fn wait_readable(replicas: &[Replica], idle: bool) {
    if replicas.iter().any(|replica| replica.peer.buffered()) {
        return;
    }
    let mut fds: Vec<libc::pollfd> = replicas
        .iter()
        .filter(|replica| replica.active())
        .map(|replica| libc::pollfd {
            fd: replica.peer.stream.as_raw_fd(),
            events: if replica.pending() {
                libc::POLLIN | libc::POLLOUT
            } else {
                libc::POLLIN
            },
            revents: 0,
        })
        .collect();
    if fds.is_empty() {
        std::thread::sleep(Duration::from_millis(1));
        return;
    }
    let timeout = if idle { 2 } else { 1 };
    // SAFETY: fds is a valid array of pollfd for the duration of the call.
    unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinity_keeps_a_key_until_load_moves_it() {
        let mut affinity = Affinity::new(2);
        let loads = [Some(0), Some(0), Some(0), Some(0)];
        let first = affinity.pick(7, &loads).unwrap();
        assert_eq!(affinity.pick(7, &loads), Some(first));
        // Different keys spread over the replicas.
        let picks: std::collections::HashSet<_> = (0..64)
            .map(|key| affinity.pick(key, &loads).unwrap())
            .collect();
        assert_eq!(picks.len(), 4);
        // Within the slack the key stays; past it, it moves to the least busy replica and
        // stays there.
        let mut busy = [Some(1); 4];
        busy[first] = Some(3);
        assert_eq!(affinity.pick(7, &busy), Some(first));
        busy[first] = Some(4);
        let moved = affinity.pick(7, &busy).unwrap();
        assert_ne!(moved, first);
        assert_eq!(affinity.pick(7, &loads), Some(moved));
        // A replica that stops serving gets no requests.
        let mut down = loads;
        down[moved] = None;
        assert_ne!(affinity.pick(7, &down), Some(moved));
        assert_eq!(affinity.pick(7, &[None, None]), None);
    }

    #[test]
    fn conversation_keys_follow_sessions_and_opening_messages() {
        let request = |messages: &[(&str, &str)], extra: Option<(&str, serde_json::Value)>| {
            let mut request: ConversationRequest = serde_json::from_value(serde_json::json!({
                "id": "a", "model": "m", "messages": [], "tools": [], "max_output_tokens": null,
                "temperature": 1.0, "top_p": 1.0, "top_k": null, "seed": null, "logprobs": false,
                "top_logprobs": 0, "prompt_logprobs": false, "routed_dp_rank": null, "stop": [],
                "reasoning_effort": null, "extra": {}
            }))
            .unwrap();
            request.messages = messages
                .iter()
                .map(|(role, text)| serde_json::json!({"role": role, "content": text}))
                .collect();
            if let Some((name, value)) = extra {
                request.extra.insert(name.to_owned(), value);
            }
            request
        };
        // Three turns with a system message, and three without one, keep their key.
        let turns = |opening: &[(&'static str, &'static str)]| {
            let mut messages = opening.to_vec();
            let mut keys = vec![conversation_key(&request(&messages, None))];
            for (answer, question) in [("reply 1", "more 1"), ("reply 2", "more 2")] {
                messages.extend([("assistant", answer), ("user", question)]);
                keys.push(conversation_key(&request(&messages, None)));
            }
            keys
        };
        let with_system = turns(&[("system", "be brief"), ("user", "hello")]);
        assert!(with_system.iter().all(|key| *key == with_system[0]));
        let without_system = turns(&[("user", "hello")]);
        assert!(without_system.iter().all(|key| *key == without_system[0]));
        assert_ne!(with_system[0], without_system[0]);
        // Another first question is another conversation.
        assert_ne!(
            without_system[0],
            conversation_key(&request(&[("user", "other")], None))
        );
        assert_ne!(
            with_system[0],
            conversation_key(&request(&[("system", "be brief"), ("user", "other")], None))
        );
        let session = Some(("prompt_cache_key", serde_json::json!("s1")));
        assert_eq!(
            conversation_key(&request(&[("user", "x")], session.clone())),
            conversation_key(&request(&[("user", "y")], session))
        );
        let user = Some(("metadata", serde_json::json!({"user_id": "u"})));
        assert_eq!(
            conversation_key(&request(&[("user", "x")], user.clone())),
            conversation_key(&request(&[("user", "y")], user))
        );
    }

    #[test]
    fn the_outbox_never_blocks_and_keeps_frames_whole() {
        let (router_end, replica_end) = std::os::unix::net::UnixStream::pair().unwrap();
        let mut replica = Replica::new(Peer::new(router_end, 1, 0).unwrap());
        let large = vec![7u8; 8 << 20];
        replica.queue(INGRESS, &large);
        replica.queue(INGRESS, b"small");
        // The other end reads nothing yet: the flush writes what the socket takes and returns.
        let started = Instant::now();
        replica.flush().unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(replica.pending());
        let mut reader = Peer::new(replica_end, 0, 0).unwrap();
        let mut frames = Vec::new();
        while frames.len() < 2 {
            replica.flush().unwrap();
            if let Some(frame) = reader.try_frame().unwrap() {
                frames.push(frame);
            }
        }
        assert!(!replica.pending());
        assert_eq!(
            (frames[0].header.seq, frames[0].payload.len()),
            (1, large.len())
        );
        assert!(frames[0].payload.iter().all(|byte| *byte == 7));
        assert_eq!(
            (frames[1].header.seq, &frames[1].payload[..]),
            (2, &b"small"[..])
        );
    }
}
