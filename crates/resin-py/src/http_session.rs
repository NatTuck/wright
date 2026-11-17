//! Shared request lifecycle for local serving and routed attention-DP drivers.
use crate::http_distributed::{HttpEvents, HttpIngress};
use crate::http_logprobs::Vocabulary;
use crate::http_output::OutputScores;
use crate::http_stop::StopFilter;
use crate::logging::HTTP;
use crate::recipe::{self, NativeRecipe, NativeResponseParser, RenderedPrompt, SharedParser};
use crate::recipe_events;
use crate::recipe_types::{Message, RenderOptions, Tool};
use crate::{CompletionReason, Engine, RequestLease, SamplingParams, TokenEvent};
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pythonize::pythonize;
use resin_core::Phase;
use resin_core::engine::StepReport;
use resin_http::{
    EVENT_QUEUE_CAPACITY, FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob,
    PromptProgress,
};
use resin_recipe::{AttributedEvent, Event as ParsedEvent, FinishReason as ParserFinishReason};
use std::{collections::HashMap, sync::Mutex, time::Instant};
use tokio::sync::mpsc;

fn into_http(event: ParsedEvent) -> GenerationEvent {
    match event {
        ParsedEvent::ReasoningDelta { delta } => GenerationEvent::ReasoningDelta(delta),
        ParsedEvent::TextDelta { delta } => GenerationEvent::TextDelta {
            delta,
            logprobs: Vec::new(),
        },
        ParsedEvent::ToolCallStart { index, id, name } => {
            GenerationEvent::ToolCallStart { index, id, name }
        }
        ParsedEvent::ToolCallArgumentsDelta { index, delta } => {
            GenerationEvent::ToolCallArgumentsDelta { index, delta }
        }
    }
}

enum Parser {
    Native(SharedParser),
    Python {
        object: Py<PyAny>,
        finish: Option<Py<PyAny>>,
    },
}

impl Parser {
    fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<Self> {
        if let Ok(native) = object.bind(py).cast::<NativeResponseParser>() {
            return Ok(Self::Native(native.borrow().inner.clone()));
        }
        let finish = optional_method(py, &object, "finish")?;
        Ok(Self::Python { object, finish })
    }

    fn token_channel(&self, id: u32) -> Option<&'static str> {
        match self {
            Self::Native(parser) => parser
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .token_channel(id),
            Self::Python { .. } => None,
        }
    }

    fn feed(&self, ids: &[u32]) -> PyResult<Vec<AttributedEvent>> {
        match self {
            Self::Native(parser) => recipe::feed(parser, ids),
            Self::Python { object, .. } => Python::attach(|py| {
                recipe_events::from_python(&object.bind(py).call_method1("feed", (ids,))?)
            }),
        }
    }

    fn finish(&self, reason: ParserFinishReason) -> PyResult<Vec<AttributedEvent>> {
        match self {
            Self::Native(parser) => recipe::finish(parser, reason),
            Self::Python { finish, .. } => Python::attach(|py| {
                let Some(method) = finish else {
                    return Ok(Vec::new());
                };
                let reason = match reason {
                    ParserFinishReason::Stop => "stop",
                    ParserFinishReason::Length => "length",
                };
                recipe_events::from_python(&method.bind(py).call1((reason,))?)
            }),
        }
    }
}

fn optional_method(py: Python<'_>, object: &Py<PyAny>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    match object.bind(py).getattr(name) {
        Ok(method) if method.is_callable() => Ok(Some(method.unbind())),
        Ok(_) => Err(PyValueError::new_err(format!(
            "parser.{name} must be callable"
        ))),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Coalesce protocol-neutral output across steps, without delaying parsing or stops.
#[derive(Default)]
struct StreamOutput {
    events: Vec<GenerationEvent>,
    scores: Vec<GeneratedTokenLogprob>,
    tokens: u32,
    sent_output: bool,
}

impl StreamOutput {
    fn has_output(&self) -> bool {
        self.events.iter().any(|event| match event {
            GenerationEvent::TextDelta { delta, .. }
            | GenerationEvent::ReasoningDelta(delta)
            | GenerationEvent::ToolCallArgumentsDelta { delta, .. } => !delta.is_empty(),
            GenerationEvent::ToolCallStart { .. } => true,
            _ => false,
        })
    }

    fn ready(&self, interval: u32, finished: bool) -> bool {
        finished
            || self.tokens >= interval
            || (!self.sent_output && self.has_output())
            // Bound noncoalescible lanes even with a large configured interval.
            || self.events.len() >= EVENT_QUEUE_CAPACITY / 2
    }

    fn flush(&mut self, sender: &mpsc::Sender<GenerationEvent>) -> bool {
        if self.events.is_empty() && self.scores.is_empty() {
            return true;
        }
        let has_output = self.has_output();
        let delivered = self
            .events
            .drain(..)
            .all(|event| sender.try_send(event).is_ok())
            && (self.scores.is_empty()
                || sender
                    .try_send(GenerationEvent::Logprobs(std::mem::take(&mut self.scores)))
                    .is_ok());
        self.sent_output |= has_output && delivered;
        self.tokens = 0;
        delivered
    }
}

struct ActiveRequest {
    http_id: String,
    /// The request's name in the log: the client's request id, else `http_id`.
    log_id: String,
    lease: Py<RequestLease>,
    parser: Parser,
    stops: StopFilter,
    scores: Option<OutputScores>,
    prompt_logprobs: bool,
    prompt_ids: Vec<u32>,
    events: mpsc::Sender<GenerationEvent>,
    started: Instant,
    first_token_at: Option<Instant>,
    input_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
    last_source: u32,
    stream: StreamOutput,
    admitted_at: Instant,
    /// Prompt tokens computed so far and the launch time attributed to them.
    prefill_tokens: u32,
    prefill_seconds: f64,
    prefill_logged: bool,
    /// This request's tokens per second in each decode step that gave it tokens.
    step_rates: Vec<f64>,
    /// The client asked for `PromptProgress` events.
    return_progress: bool,
    /// The `processed` count of the last progress event sent.
    progress_sent: usize,
    /// The prompt tokens in the KV cache after this pass's newest prefill step, not yet sent.
    progress_pending: Option<usize>,
}

/// Event queue slots a progress event leaves free: progress is dropped before it can delay
/// output or make a client look slow.
const PROGRESS_RESERVE: usize = 8;

impl ActiveRequest {
    /// Send a `PromptProgress` event for `processed` prompt tokens. The count never goes back
    /// (a prefix restore can recompute some cached tokens) and stops after the first token.
    fn send_progress(&mut self, processed: usize) {
        if !self.return_progress
            || self.first_token_at.is_some()
            || self.events.capacity() <= PROGRESS_RESERVE
        {
            return;
        }
        let processed = processed.max(self.progress_sent).min(self.input_tokens);
        self.progress_sent = processed;
        let _ = self
            .events
            .try_send(GenerationEvent::PromptProgress(PromptProgress {
                total: self.input_tokens,
                cache: self.cached_tokens,
                processed,
                time_ms: self.admitted_at.elapsed().as_millis() as u64,
            }));
    }

    fn log_prefill_done(&mut self) {
        self.prefill_logged = true;
        let computed = self.input_tokens.saturating_sub(self.cached_tokens);
        log::info!(
            target: HTTP,
            "{}: prefill done: {computed} tokens computed ({} cached) in {:.1} ms ({} tok/s); first token {:.1} ms after admission",
            self.log_id,
            self.cached_tokens,
            self.prefill_seconds * 1e3,
            describe_rate(rate(self.prefill_tokens as f64, self.prefill_seconds)),
            self.admitted_at.elapsed().as_secs_f64() * 1e3,
        );
    }

    fn log_finished(&self, reason: FinishReason, stop_sequence: Option<&str>) {
        let generation = self
            .first_token_at
            .map_or(0.0, |first| first.elapsed().as_secs_f64());
        let mut rates = self.step_rates.clone();
        rates.sort_by(f64::total_cmp);
        let steps = match (rates.first(), rates.last()) {
            (Some(min), Some(max)) => format!(
                "; per step min {} / median {} / max {} tok/s",
                describe_rate(*min),
                describe_rate(rates[rates.len() / 2]),
                describe_rate(*max)
            ),
            _ => String::new(),
        };
        let reason = match reason {
            FinishReason::Stop => match stop_sequence {
                Some(sequence) => format!("stop sequence {sequence:?}"),
                None => "stop".to_owned(),
            },
            FinishReason::Length => "length".to_owned(),
            FinishReason::Cancelled => "cancelled".to_owned(),
        };
        log::info!(
            target: HTTP,
            "{}: finished ({reason}): {} tokens generated in {:.2} s ({} tok/s{steps}); {:.2} s since admission, first token after {:.1} ms",
            self.log_id,
            self.output_tokens,
            generation,
            describe_rate(rate(self.output_tokens as f64, generation)),
            self.admitted_at.elapsed().as_secs_f64(),
            self.first_token_at.map_or(0.0, |first| {
                first.duration_since(self.admitted_at).as_secs_f64() * 1e3
            }),
        );
    }

    fn log_cancelled(&self, why: &str) {
        log::info!(
            target: HTTP,
            "{}: cancelled after {} tokens: {why}",
            self.log_id,
            self.output_tokens
        );
    }
    fn emit(&mut self, event: AttributedEvent, output: &mut Vec<GenerationEvent>) {
        if let Some(scores) = &mut self.scores {
            for event in scores.push(event) {
                push_event(output, event);
            }
        } else {
            push_event(output, into_http(event.event));
        }
    }

    fn parsed(
        &mut self,
        event: AttributedEvent,
        output: &mut Vec<GenerationEvent>,
    ) -> PyResult<()> {
        validate_spans(&event, self.output_tokens, self.scores.as_ref())?;
        if event
            .spans
            .first()
            .is_some_and(|span| span.token < self.last_source)
        {
            return Err(PyValueError::new_err(
                "response-parser token spans moved backwards",
            ));
        }
        if let Some(last) = event.spans.last() {
            self.last_source = last.token;
        }
        for event in self.stops.push(event) {
            self.emit(event, output);
        }
        Ok(())
    }

    fn flush_stops(&mut self, output: &mut Vec<GenerationEvent>) {
        for event in self.stops.finish() {
            self.emit(event, output);
        }
    }
}

fn rate(tokens: f64, seconds: f64) -> f64 {
    if seconds > 0.0 { tokens / seconds } else { 0.0 }
}

fn describe_rate(rate: f64) -> String {
    if rate >= 100.0 {
        format!("{rate:.0}")
    } else {
        format!("{rate:.1}")
    }
}

/// Take the committed steps' reports: the per-request prefill and decode figures the
/// lifecycle lines report, `PromptProgress` events for the requests that asked for them, and
/// (with `log`) one line per step naming the requests it ran. The reports come from host data
/// the Engine keeps at each commit, so none of this touches the step loop.
fn apply_steps(active: &mut HashMap<u64, ActiveRequest>, reports: Vec<StepReport>, log: bool) {
    let mut progress = false;
    for report in reports {
        let seconds = report.elapsed.as_secs_f64();
        let names = if !log {
            String::new()
        } else {
            report
                .rows
                .iter()
                .map(|row| {
                    active.get(&row.request_id).map_or_else(
                        || format!("#{}", row.request_id),
                        |request| request.log_id.clone(),
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        match report.phase {
            Phase::Prefill => {
                let tokens = report
                    .rows
                    .iter()
                    .map(|row| row.scheduled_tokens)
                    .sum::<u32>()
                    .max(1);
                if log {
                    log::debug!(
                    target: HTTP,
                    "step {} prefill: {} rows [{names}]: {tokens} tokens in {:.1} ms ({} tok/s)",
                    report.step,
                    report.rows.len(),
                    seconds * 1e3,
                    describe_rate(rate(f64::from(tokens), seconds)),
                    );
                }
                for row in &report.rows {
                    let Some(request) = active.get_mut(&row.request_id) else {
                        continue;
                    };
                    request.prefill_tokens += row.scheduled_tokens;
                    request.prefill_seconds +=
                        seconds * f64::from(row.scheduled_tokens) / f64::from(tokens);
                    if request.return_progress {
                        request.progress_pending =
                            Some((row.token_start + row.scheduled_tokens) as usize);
                        progress = true;
                    }
                    if log && (row.sampled_tokens > 0 || row.finished) && !request.prefill_logged {
                        request.log_prefill_done();
                    }
                }
            }
            Phase::Decode => {
                let rows = report.rows.len() as u32;
                let tokens = report
                    .rows
                    .iter()
                    .map(|row| row.sampled_tokens)
                    .sum::<u32>();
                if log && report.draft_tokens > 0 {
                    let drafted = rows * report.draft_tokens;
                    let accepted = tokens.saturating_sub(rows);
                    log::info!(
                        target: HTTP,
                        "step {} verify: {rows} rows [{names}]: {accepted}/{drafted} drafts accepted ({:.0}%), {tokens} tokens in {:.1} ms ({} tok/s)",
                        report.step,
                        100.0 * f64::from(accepted) / f64::from(drafted.max(1)),
                        seconds * 1e3,
                        describe_rate(rate(f64::from(tokens), seconds)),
                    );
                } else if log {
                    log::info!(
                        target: HTTP,
                        "step {} decode: {rows} rows [{names}]: {tokens} tokens in {:.1} ms ({} tok/s)",
                        report.step,
                        seconds * 1e3,
                        describe_rate(rate(f64::from(tokens), seconds)),
                    );
                }
                for row in &report.rows {
                    if row.sampled_tokens == 0 || seconds <= 0.0 {
                        continue;
                    }
                    if let Some(request) = active.get_mut(&row.request_id) {
                        request
                            .step_rates
                            .push(f64::from(row.sampled_tokens) / seconds);
                    }
                }
            }
        }
    }
    // One event per request for this pass: its newest step.
    if progress {
        for request in active.values_mut() {
            if let Some(processed) = request.progress_pending.take() {
                request.send_progress(processed);
            }
        }
    }
}

pub(crate) struct Session {
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    stop_token_ids: Vec<u32>,
    vocabulary: Vocabulary,
    active: HashMap<u64, ActiveRequest>,
    stream_interval: u32,
    pub(crate) metrics: Option<resin_http::MetricsEndpoint>,
    emit_first_token: bool,
    /// With [`Session::render_aside`]: the thread that renders admitted jobs.
    renderer: Option<Renderer>,
}

/// A thread that renders a session's jobs, so that a long prompt's tokenization does not hold the
/// Engine's loop: the loop submits what it has rendered
/// ([`Session::poll_rendered`]).
struct Renderer {
    jobs: std::sync::mpsc::Sender<GenerationJob>,
    // (a Mutex: a replica's session lives in a pyclass, which must be Sync)
    done: Mutex<std::sync::mpsc::Receiver<Rendered>>,
    pending: usize,
}

impl Renderer {
    fn new(py: Python<'_>, recipe: &Py<PyAny>, stop_token_ids: &[u32]) -> PyResult<Self> {
        let (jobs, inbox) = std::sync::mpsc::channel::<GenerationJob>();
        let (outbox, done) = std::sync::mpsc::channel::<Rendered>();
        let recipe = recipe.clone_ref(py);
        let stops = stop_token_ids.to_vec();
        std::thread::Builder::new()
            .name("wright-render".into())
            .spawn(move || {
                for job in inbox {
                    let rendered = if job.events.is_closed() {
                        Ok(None)
                    } else {
                        Python::attach(|py| render(py, &recipe, &stops, &job))
                    };
                    if outbox.send((job, rendered)).is_err() {
                        break; // the session is gone
                    }
                }
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!("starting the render thread: {error}"))
            })?;
        Ok(Self {
            jobs,
            done: Mutex::new(done),
            pending: 0,
        })
    }
}

impl Session {
    pub fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
    ) -> PyResult<Self> {
        if stream_interval == 0 {
            return Err(PyValueError::new_err(
                "stream_interval must be a positive integer",
            ));
        }
        let mut state = engine.bind(py).borrow_mut();
        if state.closed() || state.poisoned() {
            return Err(PyRuntimeError::new_err(
                "the inference engine is closed or poisoned",
            ));
        }
        state.enable_step_reports();
        drop(state);
        let vocabulary = Vocabulary::from_recipe(py, &recipe)?;
        let stop_token_ids = if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
            native.borrow().resolved_stop_token_ids()?.to_vec()
        } else {
            recipe.bind(py).getattr("stop_token_ids")?.extract()?
        };
        Ok(Self {
            engine,
            recipe,
            stop_token_ids,
            vocabulary,
            active: HashMap::new(),
            stream_interval,
            metrics: None,
            renderer: None,
            emit_first_token: false,
        })
    }
    /// Requests in the session: in the Engine, and admitted but still rendering.
    pub fn active_count(&self) -> usize {
        self.active.len() + self.rendering()
    }
    /// Requests in the Engine.
    pub fn submitted_count(&self) -> usize {
        self.active.len()
    }
    /// Admitted jobs that the render thread has not given back yet.
    pub fn rendering(&self) -> usize {
        self.renderer.as_ref().map_or(0, |r| r.pending)
    }
    /// Render admitted jobs on a thread of their own from now on (`admit` returns at once; the loop calls
    /// [`Session::poll_rendered`]).
    pub fn render_aside(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new(py, &self.recipe, &self.stop_token_ids)?);
        }
        Ok(())
    }
    /// Submit every job that the render thread has rendered.
    pub fn poll_rendered(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let done = renderer
            .done
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        while let Ok(rendered) = done.try_recv() {
            renderer.pending -= 1;
            submit(&self.engine, rendered, &mut self.active);
        }
    }
    pub fn admit(&mut self, job: GenerationJob) {
        if let Some(metrics) = &self.metrics {
            metrics.observe_queue_time(job.enqueued_at.elapsed().as_secs_f64());
        }
        if let Some(renderer) = self.renderer.as_mut() {
            if job.events.is_closed() {
                return;
            }
            match renderer.jobs.send(job) {
                Ok(()) => {
                    renderer.pending += 1;
                    return;
                }
                // the render thread stopped (it does not, but render here then)
                Err(std::sync::mpsc::SendError(job)) => {
                    self.renderer = None;
                    return self.admit(job);
                }
            }
        }
        admit(
            &self.engine,
            &self.recipe,
            &self.stop_token_ids,
            job,
            &mut self.active,
        );
    }
    pub fn cancel_disconnected(&mut self) -> Result<(), String> {
        cancel_disconnected(&self.engine, &mut self.active)
    }
    pub fn process(&mut self, tokens: Vec<TokenEvent>) -> Result<(), String> {
        let reports = Python::attach(|py| self.engine.bind(py).borrow_mut().take_step_reports());
        if !reports.is_empty() {
            let log = log::log_enabled!(target: HTTP, log::Level::Info);
            apply_steps(&mut self.active, reports, log);
        }
        process(
            &self.engine,
            &self.vocabulary,
            &mut self.active,
            self.stream_interval,
            self.metrics.as_ref(),
            self.emit_first_token,
            tokens,
        )
    }
    pub fn cancel(&mut self, ids: &[String]) -> PyResult<()> {
        let remove = self
            .active
            .iter()
            .filter_map(|(&id, request)| ids.contains(&request.http_id).then_some(id))
            .collect::<Vec<_>>();
        for id in remove {
            let request = self.active.remove(&id).expect("active id");
            request.log_cancelled("by the frontend");
            Python::attach(|py| {
                self.engine
                    .bind(py)
                    .borrow_mut()
                    .cancel(py, request.lease.bind(py).borrow())
            })?;
        }
        Ok(())
    }
    pub fn close(&mut self) -> PyResult<()> {
        let ids = self
            .active
            .values()
            .map(|request| request.http_id.clone())
            .collect::<Vec<_>>();
        self.cancel(&ids)
    }
}
fn process(
    engine: &Py<Engine>,
    vocabulary: &Vocabulary,
    active: &mut HashMap<u64, ActiveRequest>,
    stream_interval: u32,
    metrics: Option<&resin_http::MetricsEndpoint>,
    emit_first_token: bool,
    token_events: Vec<TokenEvent>,
) -> Result<(), String> {
    let mut grouped = HashMap::<u64, Vec<crate::TokenEvent>>::new();
    for event in token_events {
        grouped.entry(event.request_id).or_default().push(event);
    }
    Python::attach(|py| {
        for (&id, request) in active.iter() {
            if request.lease.bind(py).borrow().done() {
                grouped.entry(id).or_default();
            }
        }
    });
    if grouped.is_empty() {
        return Ok(());
    }
    for (request_id, tokens) in grouped {
        let Some(request) = active.get_mut(&request_id) else {
            continue;
        };
        if !tokens.is_empty() && request.first_token_at.is_none() {
            request.first_token_at = Some(Instant::now());
            if let Some(metrics) = metrics {
                metrics.observe_first_token(request.started.elapsed().as_secs_f64());
            } else if emit_first_token {
                let _ = request.events.try_send(GenerationEvent::FirstToken);
            }
        }
        let mut finished = tokens.is_empty().then_some(CompletionReason::Length);
        let mut scores = std::mem::take(&mut request.stream.scores);
        let parsed = (|| -> PyResult<Vec<GenerationEvent>> {
            let mut output = std::mem::take(&mut request.stream.events);
            if tokens.is_empty() {
                for event in request.parser.finish(ParserFinishReason::Length)? {
                    request.parsed(event, &mut output)?;
                }
                request.flush_stops(&mut output);
            }
            for token in tokens {
                request.output_tokens += 1;
                request.stream.tokens = request.stream.tokens.saturating_add(1);
                let channel = if token.logprobs.is_none() {
                    None
                } else {
                    request.parser.token_channel(token.token_id)
                };
                if let Some(logprobs) = &token.logprobs {
                    let score =
                        vocabulary.generated(token.token_id, token.index, channel, logprobs);
                    if score.token.bytes.is_some()
                        && !matches!(channel, Some("control" | "reasoning" | "tool"))
                        && let Some(output_scores) = &mut request.scores
                    {
                        output_scores.insert(score.clone());
                    }
                    scores.push(score);
                }
                let events = request.parser.feed(&[token.token_id])?;
                for event in events {
                    request.parsed(event, &mut output)?;
                    if request.stops.stopped() {
                        break;
                    }
                }
                if request.stops.stopped() {
                    finished = Some(CompletionReason::Stop);
                    Python::attach(|py| {
                        engine
                            .bind(py)
                            .borrow_mut()
                            .cancel(py, request.lease.bind(py).borrow())
                    })?;
                    break;
                }
                if let Some(reason) = token.finish_reason {
                    finished = Some(reason);
                    for event in request.parser.finish(match reason {
                        CompletionReason::Stop => ParserFinishReason::Stop,
                        CompletionReason::Length => ParserFinishReason::Length,
                    })? {
                        request.parsed(event, &mut output)?;
                    }
                    request.flush_stops(&mut output);
                    if request.stops.stopped() {
                        finished = Some(CompletionReason::Stop);
                    }
                    break;
                }
            }
            if finished.is_some()
                && let Some(scores) = &mut request.scores
            {
                for event in scores.finish() {
                    push_event(&mut output, event);
                }
            }
            Ok(output)
        })();
        let delivered = match parsed {
            Ok(events) => {
                request.stream.events = events;
                request.stream.scores = scores;
                !request.stream.ready(stream_interval, finished.is_some())
                    || request.stream.flush(&request.events)
            }
            Err(error) => {
                let message = format_python_error("response parser failed", error);
                log::warn!(target: HTTP, "{}: {message}", request.log_id);
                let _ = request.events.try_send(GenerationEvent::Error { message });
                false
            }
        };
        if !delivered {
            // A full output queue belongs to this client only. Never wait
            // for it from the one inference actor and stall every request.
            request.log_cancelled("the client is not reading its output");
            Python::attach(|py| {
                engine
                    .bind(py)
                    .borrow_mut()
                    .cancel(py, request.lease.bind(py).borrow())
            })
            .map_err(|error| format_python_error("request cancellation failed", error))?;
            active.remove(&request_id);
            continue;
        }
        if let Some(reason) = finished {
            if let Some(sequence) = request.stops.matched() {
                let _ = request
                    .events
                    .try_send(GenerationEvent::StopSequence(sequence.to_owned()));
            }
            if request.prompt_logprobs {
                let scores = Python::attach(|py| request.lease.bind(py).borrow().prompt_logprobs())
                    .unwrap_or_default()
                    .into_iter()
                    .zip(&request.prompt_ids)
                    .map(|(score, &id)| {
                        score.map(|score| vocabulary.token(id, score.inner.logprob))
                    })
                    .collect();
                let _ = request
                    .events
                    .try_send(GenerationEvent::PromptLogprobs(scores));
            }
            let reason = match reason {
                CompletionReason::Stop => FinishReason::Stop,
                CompletionReason::Length => FinishReason::Length,
            };
            if let Some(metrics) = metrics {
                metrics.observe_completed(request.started.elapsed().as_secs_f64());
            }
            if log::log_enabled!(target: HTTP, log::Level::Info) {
                request.log_finished(reason, request.stops.matched());
            }
            let _ = request.events.try_send(GenerationEvent::Finished {
                reason,
                input_tokens: request.input_tokens,
                cached_tokens: request.cached_tokens,
                output_tokens: request.output_tokens,
                elapsed_ms: request.started.elapsed().as_secs_f64() * 1000.0,
                generation_elapsed_ms: request
                    .first_token_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                time_to_first_token_ms: request.first_token_at.map_or(0.0, |first| {
                    first.duration_since(request.started).as_secs_f64() * 1000.0
                }),
            });
            active.remove(&request_id);
        }
    }
    Ok(())
}

/// A request rendered for the Engine: everything that its admission needs before `engine.submit`.
struct Prepared {
    token_ids: Vec<u32>,
    media: Option<Py<PyAny>>,
    parser: Parser,
    scores: Option<OutputScores>,
    stops: StopFilter,
    sampling_params: Py<SamplingParams>,
}

/// A rendered job: None when nothing is to be submitted (a tokenize-only request, answered here, or a
/// client gone before its render).
type Rendered = (GenerationJob, PyResult<Option<Prepared>>);

fn admit(
    engine: &Py<Engine>,
    recipe: &Py<PyAny>,
    stop_token_ids: &[u32],
    job: GenerationJob,
    active: &mut HashMap<u64, ActiveRequest>,
) {
    if job.events.is_closed() {
        return;
    }
    let rendered = Python::attach(|py| render(py, recipe, stop_token_ids, &job));
    submit(engine, (job, rendered), active);
}

/// The recipe's prompt, parser and sampling parameters of `job` (any thread: a native recipe tokenizes
/// without the GIL).
fn render(
    py: Python<'_>,
    recipe: &Py<PyAny>,
    stop_token_ids: &[u32],
    job: &GenerationJob,
) -> PyResult<Option<Prepared>> {
    {
        // JSON is an HTTP concern; both native and custom recipes receive the
        // same typed conversation. Native recipes stay inside Rust here.
        let messages = job
            .request
            .messages
            .iter()
            .map(|value| {
                let message = if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
                    native.borrow().message_from_wire(py, value)?
                } else {
                    Message::from_wire(py, value)?
                };
                Py::new(py, message)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tools = job
            .request
            .tools
            .iter()
            .map(|value| Py::new(py, Tool::from_wire(value)?))
            .collect::<PyResult<Vec<_>>>()?;
        let options = RenderOptions::from_wire(&serde_json::json!({
            "reasoning_effort": job.request.reasoning_effort,
            "extra": job.request.extra,
        }))?;
        let (token_ids, media, parser) = if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>()
        {
            let native = native.borrow();
            let rendered = native.render(py, messages, tools, Some(&options))?;
            let parser = Py::new(py, native.create_parser(py, &rendered)?)?.into_any();
            (rendered.token_ids, rendered.media, parser)
        } else {
            let kwargs = PyDict::new(py);
            kwargs.set_item("tools", tools)?;
            kwargs.set_item("options", Py::new(py, options)?)?;
            let rendered = recipe
                .bind(py)
                .call_method("render", (messages,), Some(&kwargs))?;
            let parser = recipe
                .bind(py)
                .call_method1("create_parser", (&rendered,))?
                .unbind();
            let prompt = rendered.cast::<RenderedPrompt>()?.borrow();
            // Custom recipes also return the native prompt container, so media
            // and token ownership cross one well-defined admission boundary.
            (
                prompt.token_ids.clone(),
                prompt.media.as_ref().map(|value| value.clone_ref(py)),
                parser,
            )
        };
        if token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "recipe.render returned an empty token_ids sequence",
            ));
        }
        if job.request.tokenize_only {
            log::info!(
                target: HTTP,
                "{}: tokenized: {} tokens",
                job.request.log_id(),
                token_ids.len()
            );
            let _ = job
                .events
                .try_send(GenerationEvent::Tokenized { token_ids });
            return Ok(None);
        }
        let parser = Parser::new(py, parser)?;
        let scores = job.request.logprobs.then(OutputScores::default);
        let extra = pythonize(py, &job.request.extra)?.unbind();
        let stops = StopFilter::new(job.request.stop.clone())?;
        let sampling_params = Py::new(
            py,
            SamplingParams::new(
                job.request
                    .max_output_tokens
                    .map(u32::try_from)
                    .transpose()
                    .map_err(|_| PyValueError::new_err("max_output_tokens must fit in a uint32"))?,
                stop_token_ids.to_vec(),
                job.request.temperature,
                job.request.top_p,
                job.request.top_k,
                job.request.seed,
                job.request.logprobs,
                job.request.top_logprobs,
                job.request.prompt_logprobs,
                0,
                Some(extra),
            )?,
        )?;
        Ok(Some(Prepared {
            token_ids,
            media,
            parser,
            scores,
            stops,
            sampling_params,
        }))
    }
}

/// Submit a rendered job to the Engine and track it (on the Engine's thread).
fn submit(
    engine: &Py<Engine>,
    (job, rendered): Rendered,
    active: &mut HashMap<u64, ActiveRequest>,
) {
    if job.events.is_closed() {
        return;
    }
    let result = rendered.and_then(|prepared| {
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        Python::attach(|py| {
            let Prepared {
                token_ids,
                media,
                parser,
                scores,
                stops,
                sampling_params,
            } = prepared;
            let prompt_ids = token_ids.clone();
            let input_tokens = token_ids.len();
            let lease = engine.bind(py).borrow_mut().submit(
                py,
                Some(token_ids),
                0,
                None,
                Some(sampling_params),
                media,
                None,
            )?;
            let lease_ref = lease.bind(py).borrow();
            let request_id = lease_ref.request_id();
            let cached_tokens = lease_ref.cached_tokens() as usize;
            drop(lease_ref);
            Ok::<_, PyErr>(Some((
                request_id,
                lease,
                parser,
                scores,
                stops,
                prompt_ids,
                input_tokens,
                cached_tokens,
            )))
        })
    });
    match result {
        Ok(None) => {}
        Ok(Some((
            request_id,
            lease,
            parser,
            scores,
            stops,
            prompt_ids,
            input_tokens,
            cached_tokens,
        ))) => {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens });
            log::info!(
                target: HTTP,
                "{}: admitted as engine request #{request_id} after {:.1} ms queued: {input_tokens} prompt tokens, {cached_tokens} cached",
                job.request.log_id(),
                job.enqueued_at.elapsed().as_secs_f64() * 1e3,
            );
            active.insert(
                request_id,
                ActiveRequest {
                    http_id: job.request.id.clone(),
                    log_id: job.request.log_id().to_owned(),
                    lease,
                    parser,
                    stops,
                    scores,
                    prompt_logprobs: job.request.prompt_logprobs,
                    prompt_ids,
                    events: job.events,
                    started: job.enqueued_at,
                    first_token_at: None,
                    input_tokens,
                    cached_tokens,
                    output_tokens: 0,
                    last_source: 0,
                    stream: StreamOutput::default(),
                    admitted_at: Instant::now(),
                    prefill_tokens: 0,
                    prefill_seconds: 0.0,
                    prefill_logged: false,
                    step_rates: Vec::new(),
                    return_progress: job.request.return_progress,
                    progress_sent: 0,
                    progress_pending: None,
                },
            );
            if let Some(request) = active.get_mut(&request_id) {
                request.send_progress(cached_tokens);
            }
        }
        Err(error) => {
            // invalid request content (rendering or engine validation) is the client's error
            let rejected = Python::attach(|py| {
                error.is_instance_of::<PyValueError>(py)
                    || error.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            });
            let message = format_python_error("request admission failed", error);
            if rejected {
                log::info!(target: HTTP, "{}: {message}", job.request.log_id());
            } else {
                log::error!(target: HTTP, "{}: {message}", job.request.log_id());
            }
            let _ = job.events.try_send(if rejected {
                GenerationEvent::Rejected { message }
            } else {
                GenerationEvent::Error { message }
            });
        }
    }
}

fn cancel_disconnected(
    engine: &Py<Engine>,
    active: &mut HashMap<u64, ActiveRequest>,
) -> Result<(), String> {
    let disconnected = active
        .iter()
        .filter_map(|(request_id, request)| request.events.is_closed().then_some(*request_id))
        .collect::<Vec<_>>();
    for request_id in disconnected {
        let request = active
            .remove(&request_id)
            .expect("request id came from the active map");
        request.log_cancelled("the client disconnected");
        Python::attach(|py| {
            engine
                .bind(py)
                .borrow_mut()
                .cancel(py, request.lease.bind(py).borrow())
        })
        .map_err(|error| format_python_error("request cancellation failed", error))?;
    }
    Ok(())
}

/// A session whose events are collected per step instead of streamed to HTTP clients: the
/// core of a DP driver's session and of a replica's (`http_replicas`).
pub(crate) struct SessionCore {
    session: Session,
    outputs: HashMap<String, mpsc::Receiver<GenerationEvent>>,
}

impl SessionCore {
    pub(crate) fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
        emit_first_token: bool,
    ) -> PyResult<Self> {
        let mut session = Session::new(py, engine, recipe, stream_interval)?;
        session.emit_first_token = emit_first_token;
        Ok(Self {
            session,
            outputs: HashMap::new(),
        })
    }

    /// Admit the requests of `ingress` routed to `rank`, then cancel its cancelled ones.
    pub(crate) fn admit(&mut self, ingress: &HttpIngress, rank: usize) -> PyResult<()> {
        for request in ingress
            .requests
            .iter()
            .filter(|request| request.routed_dp_rank.unwrap_or(0) == rank)
        {
            if self.outputs.contains_key(&request.id) {
                return Err(PyValueError::new_err("duplicate HTTP request id"));
            }
            let (events, receiver) = mpsc::channel(EVENT_QUEUE_CAPACITY);
            self.outputs.insert(request.id.clone(), receiver);
            self.session.admit(GenerationJob {
                request: request.clone(),
                events,
                enqueued_at: Instant::now(),
            });
        }
        self.cancel(&ingress.canceled)
    }

    pub(crate) fn cancel(&mut self, ids: &[String]) -> PyResult<()> {
        self.session.cancel(ids)?;
        for id in ids {
            self.outputs.remove(id);
        }
        Ok(())
    }

    /// Process one step's token events; return every request's events since the last call.
    pub(crate) fn process(&mut self, events: Vec<TokenEvent>) -> Result<HttpEvents, String> {
        self.session.process(events)?;
        let mut output = Vec::new();
        self.outputs.retain(|id, receiver| {
            let mut finished = false;
            while let Ok(event) = receiver.try_recv() {
                finished |= matches!(
                    event,
                    GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
                );
                output.push((id.clone(), event));
            }
            if !finished && receiver.is_closed() {
                // A verified block can exceed the bounded per-request queue.
                // The session cancels that lease; expose the same terminal
                // failure as an ordinary HTTP receiver reaching EOF.
                output.push((
                    id.clone(),
                    GenerationEvent::Error {
                        message: "the inference stream ended before completion".to_owned(),
                    },
                ));
                finished = true;
            }
            !finished
        });
        Ok(HttpEvents {
            events: output,
            ..HttpEvents::default()
        })
    }

    pub(crate) fn close(&mut self) -> PyResult<()> {
        self.session.close()?;
        self.outputs.clear();
        Ok(())
    }

    pub(crate) fn active_count(&self) -> usize {
        self.session.active_count()
    }

    /// See [`Session::render_aside`].
    pub(crate) fn render_aside(&mut self, py: Python<'_>) -> PyResult<()> {
        self.session.render_aside(py)
    }

    /// See [`Session::poll_rendered`].
    pub(crate) fn poll_rendered(&mut self) {
        self.session.poll_rendered()
    }

    /// See [`Session::rendering`].
    pub(crate) fn rendering(&self) -> usize {
        self.session.rendering()
    }
}

/// The same native lifecycle used by the ordinary actor, attached to one DP driver.
#[pyclass(name = "_HttpSession", module = "resin._resin")]
pub(crate) struct NativeHttpSession {
    inner: Mutex<SessionCore>,
}
#[pymethods]
impl NativeHttpSession {
    #[new]
    #[pyo3(signature = (engine, recipe: "Recipe", *, stream_interval = 1, emit_first_token = false))]
    fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
        emit_first_token: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(SessionCore::new(
                py,
                engine,
                recipe,
                stream_interval,
                emit_first_token,
            )?),
        })
    }
    #[pyo3(signature = (ingress, *, rank = 0))]
    fn admit(&self, ingress: &HttpIngress, rank: usize) -> PyResult<()> {
        self.core().admit(ingress, rank)
    }
    fn cancel(&self, ids: Vec<String>) -> PyResult<()> {
        self.core().cancel(&ids)
    }
    fn process(&self, py: Python<'_>, events: Vec<Py<TokenEvent>>) -> PyResult<HttpEvents> {
        let events = events
            .into_iter()
            .map(|event| event.bind(py).borrow().clone())
            .collect();
        self.core().process(events).map_err(PyRuntimeError::new_err)
    }
    fn close(&self) -> PyResult<()> {
        self.core().close()
    }
    #[getter]
    fn active_count(&self) -> usize {
        self.core().active_count()
    }
}

impl NativeHttpSession {
    fn core(&self) -> std::sync::MutexGuard<'_, SessionCore> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }
}

fn format_python_error(context: &str, error: PyErr) -> String {
    format!("{context}: {error}")
}

fn validate_spans(
    event: &AttributedEvent,
    tokens: usize,
    scores: Option<&OutputScores>,
) -> PyResult<()> {
    let length = match &event.event {
        ParsedEvent::TextDelta { delta }
        | ParsedEvent::ReasoningDelta { delta }
        | ParsedEvent::ToolCallArgumentsDelta { delta, .. } => delta.len(),
        ParsedEvent::ToolCallStart { .. } => 0,
    };
    if event.spans.iter().any(|span| {
        span.token as usize >= tokens
            || span.source.start >= span.source.end
            || span.text.start >= span.text.end
            || span.text.end as usize > length
            || scores
                .and_then(|scores| scores.source_len(span.token))
                .is_some_and(|length| span.source.end as usize > length)
    }) || event
        .spans
        .windows(2)
        .any(|pair| pair[0].token > pair[1].token)
    {
        return Err(PyValueError::new_err("invalid response-parser token spans"));
    }
    Ok(())
}

fn push_event(events: &mut Vec<GenerationEvent>, event: GenerationEvent) {
    match (events.last_mut(), event) {
        (
            Some(GenerationEvent::TextDelta { delta, logprobs }),
            GenerationEvent::TextDelta {
                delta: next,
                logprobs: scores,
            },
        ) => {
            delta.push_str(&next);
            logprobs.extend(scores);
        }
        (Some(GenerationEvent::ReasoningDelta(delta)), GenerationEvent::ReasoningDelta(next)) => {
            delta.push_str(&next);
        }
        (
            Some(GenerationEvent::ToolCallArgumentsDelta { index, delta }),
            GenerationEvent::ToolCallArgumentsDelta {
                index: next_index,
                delta: next,
            },
        ) if *index == next_index => delta.push_str(&next),
        (_, event) => events.push(event),
    }
}
