//! The gate of one card that a rank of a tensor-parallel model and a one-card replica share
//! (`resin_gaudi::card_gate` on the loaded runtime): the native executors and the replica server call it without
//! Python, with the GIL released while it waits.

use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use resin_gaudi::card_gate::{self, CardOps};
use resin_gaudi::glue::Glue;

use crate::gaudi::GaudiGlue;
use crate::http_replicas::StepGate;

/// `resin_models_gaudi.synapse.card_gate.CardGate` builds it: three event slots from `first` (the replica's two
/// timed), the replica's `share` of the card's time while both models have work, and the TP model's `holdoff`.
#[pyclass(module = "resin._resin", frozen)]
pub(crate) struct CardGate {
    pub(crate) inner: Arc<card_gate::CardGate<Glue>>,
}

/// Replica steps in the TP model's idle time and in its share, and TP steps that waited for a replica step.
#[pyclass(module = "resin._resin", frozen, get_all)]
pub(crate) struct GateCounts {
    idle_steps: u64,
    shared_steps: u64,
    tp_waits: u64,
}

impl CardGate {
    /// A TP step starts on this card (the native executors call it on their own thread and context).
    pub(crate) fn begin(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.tp_begin())
            .map_err(PyRuntimeError::new_err)
    }

    /// The TP step's work is on the device.
    pub(crate) fn end(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.tp_end())
            .map_err(PyRuntimeError::new_err)
    }
}

impl<O: CardOps + Send + Sync> StepGate for card_gate::CardGate<O> {
    fn acquire(&self, timeout: Duration) -> Result<bool, String> {
        card_gate::CardGate::acquire(self, timeout)
    }

    fn release(&self) -> Result<(), String> {
        card_gate::CardGate::release(self)
    }
}

#[pymethods]
impl CardGate {
    #[new]
    #[pyo3(signature = (glue, first, share=0.0, holdoff=card_gate::HOLDOFF))]
    fn new(glue: PyRef<'_, GaudiGlue>, first: i32, share: f64, holdoff: f64) -> PyResult<Self> {
        let gate = card_gate::CardGate::new(glue.glue, first, share, holdoff)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(gate),
        })
    }

    fn tp_begin(&self, py: Python<'_>) -> PyResult<()> {
        self.begin(py)
    }

    fn tp_end(&self, py: Python<'_>) -> PyResult<()> {
        self.end(py)
    }

    /// Wait up to `timeout` seconds for the replica's turn; False: not this time.
    fn acquire(&self, py: Python<'_>, timeout: f64) -> PyResult<bool> {
        let inner = &self.inner;
        let timeout = Duration::from_secs_f64(timeout.max(0.0));
        py.detach(|| inner.acquire(timeout))
            .map_err(PyRuntimeError::new_err)
    }

    /// The replica's step is on the device.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.release())
            .map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn counts(&self) -> GateCounts {
        let c = self.inner.counts();
        GateCounts {
            idle_steps: c.idle_steps,
            shared_steps: c.shared_steps,
            tp_waits: c.tp_waits,
        }
    }

    #[getter]
    fn tp_busy(&self) -> bool {
        self.inner.tp_busy()
    }

    #[getter]
    fn tp_slot(&self) -> i32 {
        self.inner.tp_slot
    }

    #[getter]
    fn replica_slot(&self) -> i32 {
        self.inner.replica_slot
    }
}
