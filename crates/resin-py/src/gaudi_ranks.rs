//! The ranks of a tensor-parallel Gaudi model over Unix sockets, without Gloo (cargo feature `gaudi`), for every
//! native step executor.
//!
//! The parent picks a socket path and a run token. Rank 0 listens on the path and the others connect
//! (`step_channel::Listener` / `connect`: the token and the same user). At startup the group exchanges the HCCL
//! unique id (`ccl_init`), agrees the smallest cache capacity (`min`) and waits at barriers. Then rank 0's
//! executor (the driver) sends each step to the others (the executor's `serve` on the followers), which make the
//! same step on their cards:
//!
//! - STEP (driver -> followers): the runner call (forward, enqueue, release, close), the plan (`BatchPlan`'s
//!   wire archive) and each row's sampling inputs, whether the step continues the last one, and the driver's
//!   draft decision (DeepSeek; GLM's driver announces a call before it decides, and sends the decision with GO);
//! - PREPARED (followers -> driver) and GO (driver -> followers) around every program run: each rank fills its
//!   tables first, and no rank makes collectives unless every rank is ready (GO carries the first failure, and a
//!   word from the driver: GLM's draft decision, which the driver makes after it announced the step).

use std::path::PathBuf;
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::BatchPlan;
use crate::gaudi::GaudiGlue;
use crate::gaudi_exec::{RANK_FAILED, decode_step, encode_step};
use crate::step_channel::frame::Header;
use crate::step_channel::socket::Peer;
use crate::step_channel::{Listener, connect, deadline, io_error, wait_frame, who};

pub(crate) const STEP: u8 = 40;
pub(crate) const PREPARED: u8 = 41;
pub(crate) const GO: u8 = 42;
const BCAST: u8 = 43;
const GATHER: u8 = 44;

/// The runner calls a STEP frame carries.
pub(crate) const OP_FORWARD: u8 = 1;
pub(crate) const OP_ENQUEUE: u8 = 2;
pub(crate) const OP_RELEASE: u8 = 3;
pub(crate) const OP_CLOSE: u8 = 4;

/// One rank's end of the group: the driver's sockets to every follower, or a follower's socket to the driver.
#[pyclass(module = "resin._resin")]
pub(crate) struct GaudiRanks {
    pub(crate) rank: u16,
    pub(crate) size: u16,
    peers: Vec<Peer>,
    seq: u64,
    timeout: f64,
}

impl GaudiRanks {
    fn header(&self, kind: u8, status: u32) -> Header {
        Header {
            kind,
            rank: self.rank,
            status,
            seq: self.seq,
        }
    }

    fn send_all(
        &self,
        py: Python<'_>,
        kind: u8,
        status: u32,
        payload: &[u8],
        what: &str,
    ) -> PyResult<()> {
        let header = self.header(kind, status);
        let until = deadline(Some(self.timeout))?.unwrap();
        let failure = py.detach(|| {
            let mut failure = None;
            for peer in &self.peers {
                if let Err(error) = peer.send(header, &[payload], until) {
                    failure.get_or_insert((who(peer), error));
                }
            }
            failure
        });
        match failure {
            Some((peer, error)) => Err(io_error(format!("sending {what} to {peer}"), error)),
            None => Ok(()),
        }
    }

    /// The next frame from peer `i`, which must be `kind` of this sequence.
    fn receive(
        &mut self,
        py: Python<'_>,
        i: usize,
        kind: u8,
        what: &str,
    ) -> PyResult<(u32, Vec<u8>)> {
        let until = deadline(Some(self.timeout))?;
        let seq = self.seq;
        let peer = &mut self.peers[i];
        let frame = wait_frame(py, peer, until, what)?;
        if frame.header.kind != kind || frame.header.seq != seq {
            return Err(PyRuntimeError::new_err(format!(
                "Qwen rank group protocol error: wanted {what} ({kind}, sequence {seq}) from {}, got kind {} \
                 sequence {}",
                who(peer),
                frame.header.kind,
                frame.header.seq
            )));
        }
        Ok((frame.header.status, frame.payload))
    }

    /// Rank 0's `payload` on every rank.
    pub(crate) fn broadcast_bytes(&mut self, py: Python<'_>, payload: &[u8]) -> PyResult<Vec<u8>> {
        self.seq += 1;
        if self.rank == 0 {
            self.send_all(py, BCAST, 0, payload, "a broadcast")?;
            return Ok(payload.to_vec());
        }
        Ok(self.receive(py, 0, BCAST, "a broadcast")?.1)
    }

    /// Every rank's `payload` on rank 0 (in rank order; None on the others).
    pub(crate) fn gather_bytes(
        &mut self,
        py: Python<'_>,
        payload: &[u8],
    ) -> PyResult<Option<Vec<Vec<u8>>>> {
        self.seq += 1;
        if self.rank != 0 {
            self.send_all(py, GATHER, 0, payload, "a gather")?;
            return Ok(None);
        }
        let mut out = vec![payload.to_vec()];
        for i in 0..self.peers.len() {
            out.push(self.receive(py, i, GATHER, "a gather")?.1);
        }
        Ok(Some(out))
    }

    // ---- the step protocol
    /// The driver: send STEP `payload` to every follower.
    pub(crate) fn send_step(&mut self, py: Python<'_>, payload: &[u8]) -> PyResult<()> {
        self.seq += 1;
        self.send_all(py, STEP, 0, payload, "a step")
    }

    /// A follower: the next STEP's payload.
    pub(crate) fn receive_step(&mut self, py: Python<'_>) -> PyResult<Vec<u8>> {
        self.seq += 1;
        let until = deadline(None)?; // a follower waits for the next step as long as the server runs
        let seq = self.seq;
        let peer = &mut self.peers[0];
        let frame = wait_frame(py, peer, until, "a step")?;
        if frame.header.kind != STEP || frame.header.seq != seq {
            return Err(PyRuntimeError::new_err(format!(
                "rank group protocol error: wanted a step (sequence {seq}) from the driver, got kind {} \
                 sequence {}",
                frame.header.kind, frame.header.seq
            )));
        }
        Ok(frame.payload)
    }

    /// Before a program run: every rank reports whether it is ready (its tables filled); the run goes ahead only
    /// when all are. Returns the first failure of any rank, and the driver's `word` (GO's status above its
    /// failure bit).
    pub(crate) fn gate(
        &mut self,
        py: Python<'_>,
        ready: Result<(), String>,
        word: u32,
    ) -> PyResult<(Result<(), String>, u32)> {
        self.seq += 1;
        let mine = ready.err();
        if self.rank != 0 {
            let message = mine.unwrap_or_default();
            let status = u32::from(!message.is_empty());
            self.send_all(py, PREPARED, status, message.as_bytes(), "PREPARED")?;
            crate::step_trace::mark("prep_sent", 0);
            let (status, payload) = self.receive(py, 0, GO, "GO")?;
            crate::step_trace::mark("go_recv", 0);
            let outcome = if status & 1 == 0 {
                Ok(())
            } else {
                Err(String::from_utf8_lossy(&payload).into())
            };
            return Ok((outcome, status >> 1));
        }
        let mut failure = mine.map(|m| format!("rank 0: {m}"));
        for i in 0..self.peers.len() {
            let (status, payload) = self.receive(py, i, PREPARED, "PREPARED")?;
            if status != 0 && failure.is_none() {
                failure = Some(format!(
                    "rank {}: {}",
                    i + 1,
                    String::from_utf8_lossy(&payload)
                ));
            }
        }
        crate::step_trace::mark("prep_all", 0);
        let message = failure.clone().unwrap_or_default();
        self.send_all(
            py,
            GO,
            u32::from(failure.is_some()) | word << 1,
            message.as_bytes(),
            "GO",
        )?;
        let outcome = match failure {
            Some(m) => Err(m),
            None => Ok(()),
        };
        Ok((outcome, word))
    }
}

#[pymethods]
impl GaudiRanks {
    /// Rank `rank` of `size`: rank 0 listens on `path` until every other rank has joined (with `token`); the
    /// others connect to it.
    #[new]
    #[pyo3(signature = (path, token, rank, size, timeout=600.0))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        token: Vec<u8>,
        rank: u16,
        size: u16,
        timeout: f64,
    ) -> PyResult<Self> {
        if size == 0 || rank >= size {
            return Err(PyValueError::new_err(format!("rank {rank} of {size}")));
        }
        let peers = if size == 1 {
            Vec::new()
        } else if rank == 0 {
            let mut listener = Listener::new(path, token, (1..size).collect())?;
            listener
                .accept(py, timeout)?
                .into_iter()
                .map(|(peer, _)| peer)
                .collect()
        } else {
            vec![connect(py, &path, &token, rank, b"", timeout)?]
        };
        Ok(Self {
            rank,
            size,
            peers,
            seq: 0,
            timeout,
        })
    }

    #[getter]
    fn rank(&self) -> u16 {
        self.rank
    }

    #[getter]
    fn size(&self) -> u16 {
        self.size
    }

    /// Rank 0's bytes on every rank (the others pass anything).
    fn broadcast(&mut self, py: Python<'_>, payload: Vec<u8>) -> PyResult<Vec<u8>> {
        self.broadcast_bytes(py, &payload)
    }

    /// The smallest `value` over the ranks, on every rank.
    fn min(&mut self, py: Python<'_>, value: i64) -> PyResult<i64> {
        let gathered = self.gather_bytes(py, &value.to_le_bytes())?;
        let smallest = gathered.map(|all| {
            all.iter()
                .map(|bytes| i64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")))
                .min()
                .expect("a rank")
        });
        let out = self.broadcast_bytes(py, &smallest.unwrap_or(0).to_le_bytes())?;
        Ok(i64::from_le_bytes(out[..8].try_into().expect("8 bytes")))
    }

    /// Wait until every rank is here.
    fn barrier(&mut self, py: Python<'_>) -> PyResult<()> {
        self.gather_bytes(py, &[])?;
        self.broadcast_bytes(py, &[])?;
        Ok(())
    }

    /// Make the group's HCCL communicator on the calling thread's current context: rank 0's unique id on every
    /// rank, then every rank's `hcclCommInitRank`.
    fn ccl_init(&mut self, py: Python<'_>, glue: &GaudiGlue) -> PyResult<()> {
        let id = if self.rank == 0 {
            glue.glue.ccl_unique_id().map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };
        let id = self.broadcast_bytes(py, &id)?;
        let (size, rank) = (i32::from(self.size), i32::from(self.rank));
        let start = Instant::now();
        py.detach(|| glue.glue.ccl_init(&id, size, rank))
            .map_err(|error| {
                PyRuntimeError::new_err(format!("{error} after {:?}", start.elapsed()))
            })
    }
}

/// A native executor's place in a rank group: the driver announces each runner call, the followers replay them.
pub(crate) struct Ranked {
    pub(crate) group: Py<GaudiRanks>,
    pub(crate) driver: bool,
    /// the driver's last enqueued step (what a continuation's feedback continues)
    pub(crate) last: Option<Py<PyAny>>,
    /// the driver's benchmark acceptance, which each call's frame carries to the followers
    pub(crate) fixed: Option<usize>,
}

impl Ranked {
    /// This process's place in `group`, or None for a group of one rank.
    pub(crate) fn of(py: Python<'_>, group: Py<GaudiRanks>) -> Option<Self> {
        let (rank, size) = {
            let g = group.borrow(py);
            (g.rank, g.size)
        };
        (size > 1).then_some(Self {
            group,
            driver: rank == 0,
            last: None,
            fixed: None,
        })
    }

    /// The driver: send this runner call to the followers, which replay it (`serve`). Returns the nanoseconds it
    /// took.
    pub(crate) fn announce(
        &self,
        py: Python<'_>,
        op: u8,
        plan: Option<&BatchPlan>,
        continues: bool,
        speculating: bool,
        ids: &[u64],
    ) -> PyResult<u64> {
        if !self.driver {
            return Ok(0);
        }
        let start = Instant::now();
        let payload = encode_step(op, continues, speculating, self.fixed, plan, ids)?;
        self.group.borrow_mut(py).send_step(py, &payload)?;
        crate::step_trace::mark("announced", i64::from(op));
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// The driver: whether `step` (what a continuation's feedback belongs to) is the last enqueued step.
    pub(crate) fn continues_last(&self, py: Python<'_>, step: &Bound<'_, PyAny>) -> bool {
        self.last
            .as_ref()
            .is_some_and(|last| step.is(last.bind(py)))
    }

    /// The driver: the call's result is the step that the next call can continue.
    pub(crate) fn led(&mut self, py: Python<'_>, out: &PyResult<Py<PyAny>>) {
        if self.driver
            && let Ok(out) = out
        {
            self.last = Some(out.clone_ref(py));
        }
    }

    /// Tell the group whether this rank is ready for its device work and go ahead only when every rank is (so
    /// no rank enters a collective alone). Returns `prepared`, or the first failure of any rank. Adds the time
    /// it took to `gate_ns`.
    pub(crate) fn gate<T>(
        &self,
        py: Python<'_>,
        prepared: PyResult<T>,
        gate_ns: &mut u64,
    ) -> PyResult<T> {
        self.gate_word(py, prepared, 0, gate_ns)
            .map(|(value, _)| value)
    }

    /// `gate`, and the driver's `word` on every rank (the driver sends it with GO; a follower's is ignored).
    pub(crate) fn gate_word<T>(
        &self,
        py: Python<'_>,
        prepared: PyResult<T>,
        word: u32,
        gate_ns: &mut u64,
    ) -> PyResult<(T, u32)> {
        let ready = prepared
            .as_ref()
            .map(|_| ())
            .map_err(|error| error.to_string());
        let start = Instant::now();
        let (outcome, word) = self.group.borrow_mut(py).gate(py, ready, word)?;
        *gate_ns += start.elapsed().as_nanos() as u64;
        match (prepared, outcome) {
            (Ok(value), Ok(())) => Ok((value, word)),
            (Err(error), _) => Err(error),
            (Ok(_), Err(message)) => {
                Err(PyRuntimeError::new_err(format!("{RANK_FAILED}{message}")))
            }
        }
    }
}

/// A native executor whose follower ranks replay the driver's runner calls (`serve`).
pub(crate) trait Replay:
    pyo3::PyClass<Frozen = pyo3::pyclass::boolean_struct::False>
{
    /// Whether a follower takes each step's result as the driver's Engine does: the previous step's after an
    /// enqueue, before a synchronous step (`DistributedWorker`: so that state the results record, such as an
    /// Engram context, and the rows' release agree on every rank).
    const HARVESTS: bool = false;
    /// The executor's rank group, when this rank is a follower.
    fn follower_group(&self, py: Python<'_>) -> Option<Py<GaudiRanks>>;
    /// The driver's draft decision for the call it replays (None: after the call).
    fn set_speculating(&mut self, _decision: Option<bool>) {}
    /// The driver's benchmark acceptance (`fixed_acceptance`), from each call's frame: a follower accepts drafts as
    /// the driver does, or its positions and states part from the plans.
    fn set_fixed(&mut self, _fixed: Option<usize>) {}
    fn replay_forward(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
    ) -> PyResult<Py<PyAny>>;
    fn replay_enqueue(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        plan: Bound<'_, BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>>;
    /// The driver's Engine took the open step's result before it released rows.
    fn replay_release(slf: &Bound<'_, Self>, py: Python<'_>, ids: &[u64]) -> PyResult<()>;
}

/// A follower: replay the driver's runner calls until it closes the group. Returns the calls replayed.
pub(crate) fn serve<E: Replay>(slf: &Bound<'_, E>, py: Python<'_>) -> PyResult<u64> {
    let group = slf.borrow().follower_group(py).ok_or_else(|| {
        PyValueError::new_err("only a follower rank serves (attach_ranks with a rank above 0)")
    })?;
    let mut last: Option<Py<PyAny>> = None;
    let mut calls = 0u64;
    loop {
        let payload = group.borrow_mut(py).receive_step(py)?;
        let frame = decode_step(&payload)?;
        crate::step_trace::mark("frame", i64::from(frame.op));
        calls += 1;
        let result = match frame.op {
            OP_FORWARD | OP_ENQUEUE => {
                let plan = frame
                    .plan
                    .ok_or_else(|| PyValueError::new_err("a step frame without a plan"))?;
                let plan = Bound::new(py, plan)?;
                let feedback = match (&last, frame.continues) {
                    (_, false) => None,
                    (Some(last), true) => Some(last.bind(py).getattr("feedback")?),
                    (None, true) => {
                        return Err(PyRuntimeError::new_err(
                            "the driver continues a step that this rank did not make",
                        ));
                    }
                };
                {
                    let mut executor = slf.borrow_mut();
                    executor.set_speculating(Some(frame.speculating));
                    executor.set_fixed(frame.fixed);
                }
                let previous = last.take();
                let harvest = |previous: Option<Py<PyAny>>| -> PyResult<()> {
                    match previous {
                        // a completion (an enqueued step), not a synchronous step's output
                        Some(previous)
                            if E::HARVESTS && previous.bind(py).hasattr("feedback")? =>
                        {
                            previous.bind(py).call_method0("result").map(drop)
                        }
                        _ => Ok(()),
                    }
                };
                let out = if frame.op == OP_FORWARD {
                    harvest(previous).and_then(|()| E::replay_forward(slf, py, plan))
                } else {
                    E::replay_enqueue(slf, py, plan, feedback)
                        .and_then(|out| harvest(previous).map(|()| out))
                };
                slf.borrow_mut().set_speculating(None);
                out.map(|out| last = Some(out))
            }
            OP_RELEASE => E::replay_release(slf, py, &frame.ids),
            OP_CLOSE => {
                crate::step_trace::flush();
                return Ok(calls);
            }
            op => {
                return Err(PyRuntimeError::new_err(format!(
                    "a step frame with an unknown call {op}"
                )));
            }
        };
        if let Err(error) = result {
            // a plan error meets every rank, and a failure at a gate every rank knows: the ranks stay in step. Any
            // other failure leaves this rank's state behind the driver's. (The exception's message, not the error's
            // display, which starts with the exception's type.)
            if !(error.is_instance_of::<PyValueError>(py)
                || error.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                || at_gate(py, &error))
            {
                return Err(error);
            }
        }
    }
}

/// Whether `error` is a gate's report of another rank's failure (`Ranked::gate`).
fn at_gate(py: Python<'_>, error: &PyErr) -> bool {
    error.value(py).to_string().starts_with(RANK_FAILED)
}
