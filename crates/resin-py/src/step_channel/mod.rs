//! The step channel: framed messages over Unix sockets between the driver of one request replica
//! (its first rank) and the replica's followers, on one host.
//!
//! The driver listens on a socket path in a private directory. Each follower connects and sends
//! HELLO with its rank and the run token; the driver checks the token and that the peer runs as
//! the same user, then removes the path. After that, either side sends frames of a kind,
//! sequence number, status and payload (see `frame`). Python gives the kinds their meaning; kind
//! 1 is HELLO. Every blocking call releases the GIL, checks for Python signals at least every
//! 100 ms and stops at its deadline. A peer that exits closes its sockets, so the other side's
//! next read or write fails within about a millisecond.

pub(crate) mod frame;
pub(crate) mod socket;

use std::io;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;

use frame::{Frame, Header};
use socket::{Peer, Wait};

const HELLO: u8 = 1;

type Received<'py> = (u8, u64, u32, Bound<'py, PyArray1<u8>>);
type Reply<'py> = (u16, u32, Bound<'py, PyArray1<u8>>);

pub(crate) fn deadline(timeout: Option<f64>) -> PyResult<Option<Instant>> {
    match timeout {
        None => Ok(None),
        Some(seconds) if seconds.is_finite() && seconds >= 0.0 => {
            Ok(Some(Instant::now() + Duration::from_secs_f64(seconds)))
        }
        Some(seconds) => Err(PyValueError::new_err(format!(
            "timeout {seconds} is not valid"
        ))),
    }
}

pub(crate) fn who(peer: &Peer) -> String {
    if peer.rank == 0 {
        format!("the driver (pid {})", peer.pid)
    } else {
        format!("rank {} (pid {})", peer.rank, peer.pid)
    }
}

pub(crate) fn io_error(context: String, error: io::Error) -> PyErr {
    let message = format!("{context}: {error}");
    match error.kind() {
        io::ErrorKind::UnexpectedEof
        | io::ErrorKind::BrokenPipe
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::NotConnected => PyConnectionError::new_err(message),
        io::ErrorKind::TimedOut => PyTimeoutError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// Wait for one frame from `peer`, without the GIL, checking signals every slice.
pub(crate) fn wait_frame(
    py: Python<'_>,
    peer: &mut Peer,
    until: Option<Instant>,
    what: &str,
) -> PyResult<Frame> {
    let mut spin = socket::spin();
    loop {
        match py.detach(|| peer.wait(std::mem::take(&mut spin))) {
            Ok(Wait::Frame(frame)) => return Ok(frame),
            Ok(Wait::Slice) => {
                py.check_signals()?;
                if until.is_some_and(|until| Instant::now() >= until) {
                    return Err(PyTimeoutError::new_err(format!(
                        "{} sent no {what} before the deadline",
                        who(peer)
                    )));
                }
            }
            Err(error) => return Err(io_error(format!("{what} from {}", who(peer)), error)),
        }
    }
}

fn expect(peer: &Peer, frame: &Frame, kind: u8, seq: u64, what: &str) -> PyResult<()> {
    let header = frame.header;
    if header.kind != kind || header.seq != seq || header.rank != peer.rank {
        return Err(PyRuntimeError::new_err(format!(
            "step channel protocol error: wanted {what} (kind {kind}, sequence {seq}) from {}, \
             got kind {} with sequence {} from rank {}",
            who(peer),
            header.kind,
            header.seq,
            header.rank
        )));
    }
    Ok(())
}

fn slices<'a>(pieces: &'a [PyReadonlyArray1<'_, u8>]) -> PyResult<Vec<&'a [u8]>> {
    pieces
        .iter()
        .map(|piece| {
            piece.as_slice().map_err(|_| {
                PyValueError::new_err("step channel payloads must be contiguous uint8 arrays")
            })
        })
        .collect()
}

fn send_frame(
    py: Python<'_>,
    peers: &[Peer],
    header: Header,
    pieces: &[PyReadonlyArray1<'_, u8>],
    timeout: f64,
    what: &str,
) -> PyResult<()> {
    let pieces = slices(pieces)?;
    let until = deadline(Some(timeout))?.unwrap();
    // Every peer gets the frame even when an earlier one fails, so that a failure reaches all
    // the others; the first failure is raised after.
    let failure = py.detach(|| {
        let mut failure = None;
        for peer in peers {
            if let Err(error) = peer.send(header, &pieces, until) {
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

/// A listening socket path that accepts peers of known ranks with the run token, then is
/// removed. The step channel's driver and the replica router (`http_replicas`) use it.
pub(crate) struct Listener {
    path: PathBuf,
    token: Vec<u8>,
    expected: Vec<u16>,
    listener: Option<UnixListener>,
}

impl Listener {
    pub(crate) fn new(path: PathBuf, token: Vec<u8>, ranks: Vec<u16>) -> PyResult<Self> {
        if token.len() < 16 {
            return Err(PyValueError::new_err(
                "the run token must have at least 16 bytes",
            ));
        }
        let mut expected = ranks;
        expected.sort_unstable();
        if expected.contains(&0) || expected.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(
                "follower ranks must be distinct and not 0",
            ));
        }
        let listener = UnixListener::bind(&path)
            .map_err(|error| io_error(format!("listening on {}", path.display()), error))?;
        listener
            .set_nonblocking(true)
            .map_err(|error| io_error("listening".into(), error))?;
        Ok(Listener {
            path,
            token,
            expected,
            listener: Some(listener),
        })
    }

    /// Wait until every expected rank has joined, then remove the socket path. The peers come
    /// back in rank order, each with the payload of its HELLO after the token.
    pub(crate) fn accept(
        &mut self,
        py: Python<'_>,
        timeout: f64,
    ) -> PyResult<Vec<(Peer, Vec<u8>)>> {
        let until = deadline(Some(timeout))?.unwrap();
        let Some(listener) = &self.listener else {
            return Err(PyRuntimeError::new_err("the step channel is not listening"));
        };
        let mut joined: Vec<(Peer, Vec<u8>)> = Vec::new();
        while joined.len() < self.expected.len() {
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    py.detach(|| std::thread::sleep(Duration::from_millis(1)));
                    py.check_signals()?;
                    if Instant::now() >= until {
                        let missing: Vec<String> = self
                            .expected
                            .iter()
                            .filter(|rank| !joined.iter().any(|(peer, _)| peer.rank == **rank))
                            .map(u16::to_string)
                            .collect();
                        return Err(PyTimeoutError::new_err(format!(
                            "ranks {} did not join the step channel",
                            missing.join(", ")
                        )));
                    }
                    continue;
                }
                Err(error) => return Err(io_error("accepting a follower".into(), error)),
            };
            let (pid, uid) = socket::peer_credentials(&stream)
                .map_err(|error| io_error("reading a follower's credentials".into(), error))?;
            if uid != socket::effective_uid() {
                return Err(PyRuntimeError::new_err(format!(
                    "process {pid} of user {uid} connected to the step channel"
                )));
            }
            let mut peer = Peer::new(stream, u16::MAX, pid)
                .map_err(|error| io_error("configuring a follower socket".into(), error))?;
            let hello = wait_frame(py, &mut peer, Some(until), "HELLO")?;
            let rank = hello.header.rank;
            let token = self.token.len();
            if hello.header.kind != HELLO
                || hello.payload.len() < token
                || hello.payload[..token] != self.token[..]
            {
                return Err(PyRuntimeError::new_err(format!(
                    "process {pid} sent a wrong HELLO to the step channel"
                )));
            }
            if !self.expected.contains(&rank) || joined.iter().any(|(peer, _)| peer.rank == rank) {
                return Err(PyRuntimeError::new_err(format!(
                    "process {pid} joined the step channel as unexpected rank {rank}"
                )));
            }
            peer.rank = rank;
            let extra = hello.payload[token..].to_vec();
            joined.push((peer, extra));
        }
        joined.sort_by_key(|(peer, _)| peer.rank);
        self.close();
        Ok(joined)
    }

    pub(crate) fn close(&mut self) {
        if self.listener.take().is_some() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.close();
    }
}

/// Connect to the listener at `path` as `rank` and send HELLO: the token, then `extra`.
pub(crate) fn connect(
    py: Python<'_>,
    path: &std::path::Path,
    token: &[u8],
    rank: u16,
    extra: &[u8],
    timeout: f64,
) -> PyResult<Peer> {
    if rank == 0 {
        return Err(PyValueError::new_err("rank 0 is the driver"));
    }
    let until = deadline(Some(timeout))?.unwrap();
    let stream = loop {
        match UnixStream::connect(path) {
            Ok(stream) => break stream,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                ) && Instant::now() < until =>
            {
                py.detach(|| std::thread::sleep(Duration::from_millis(5)));
                py.check_signals()?;
            }
            Err(error) => {
                return Err(io_error(format!("connecting to {}", path.display()), error));
            }
        }
    };
    let (pid, uid) = socket::peer_credentials(&stream)
        .map_err(|error| io_error("reading the driver's credentials".into(), error))?;
    if uid != socket::effective_uid() {
        return Err(PyRuntimeError::new_err(format!(
            "the step channel at {} belongs to process {pid} of user {uid}",
            path.display()
        )));
    }
    let peer = Peer::new(stream, 0, pid)
        .map_err(|error| io_error("configuring the step channel".into(), error))?;
    let hello = Header {
        kind: HELLO,
        rank,
        status: 0,
        seq: 0,
    };
    py.detach(|| peer.send(hello, &[token, extra], until))
        .map_err(|error| io_error("sending HELLO to the driver".into(), error))?;
    Ok(peer)
}

/// The driver end: one socket per follower, in rank order.
#[pyclass(module = "resin._resin")]
pub struct StepChannelDriver {
    listener: Listener,
    peers: Vec<Peer>,
    closed: bool,
}

impl StepChannelDriver {
    fn check_open(&self) -> PyResult<()> {
        if self.closed {
            return Err(PyConnectionError::new_err("the step channel is closed"));
        }
        Ok(())
    }
}

#[pymethods]
impl StepChannelDriver {
    /// Listen on `path` for the followers `ranks`; `token` is the secret they must send.
    #[new]
    #[pyo3(signature = (path: "str | os.PathLike[str]", token: "bytes", ranks: "Sequence[int]"))]
    fn new(path: PathBuf, token: Vec<u8>, ranks: Vec<u16>) -> PyResult<Self> {
        Ok(StepChannelDriver {
            listener: Listener::new(path, token, ranks)?,
            peers: Vec::new(),
            closed: false,
        })
    }

    /// Wait until every follower has joined, then remove the socket path.
    #[pyo3(signature = (timeout: "float") -> "None")]
    fn accept(&mut self, py: Python<'_>, timeout: f64) -> PyResult<()> {
        let joined = self.listener.accept(py, timeout)?;
        self.peers = joined.into_iter().map(|(peer, _)| peer).collect();
        Ok(())
    }

    /// The followers' ranks and process ids, in rank order.
    #[getter]
    fn followers(&self) -> Vec<(u16, i32)> {
        self.peers
            .iter()
            .map(|peer| (peer.rank, peer.pid))
            .collect()
    }

    /// Send one frame to every follower.
    #[pyo3(signature = (
        kind: "int",
        seq: "int",
        status: "int",
        pieces: "Sequence[npt.NDArray[np.uint8]]",
        timeout: "float",
        what: "str",
    ) -> "None")]
    #[allow(clippy::too_many_arguments)]
    fn send(
        &self,
        py: Python<'_>,
        kind: u8,
        seq: u64,
        status: u32,
        pieces: Vec<PyReadonlyArray1<'_, u8>>,
        timeout: f64,
        what: &str,
    ) -> PyResult<()> {
        self.check_open()?;
        let header = Header {
            kind,
            rank: 0,
            status,
            seq,
        };
        send_frame(py, &self.peers, header, &pieces, timeout, what)
    }

    /// One frame of `kind` and `seq` from every follower, in rank order: (rank, status, payload).
    #[pyo3(signature = (kind: "int", seq: "int", timeout: "float | None", what: "str")
        -> "list[tuple[int, int, npt.NDArray[np.uint8]]]")]
    fn gather<'py>(
        &mut self,
        py: Python<'py>,
        kind: u8,
        seq: u64,
        timeout: Option<f64>,
        what: &str,
    ) -> PyResult<Vec<Reply<'py>>> {
        self.check_open()?;
        let until = deadline(timeout)?;
        let mut replies = Vec::with_capacity(self.peers.len());
        for peer in &mut self.peers {
            let frame = wait_frame(py, peer, until, what)?;
            expect(peer, &frame, kind, seq, what)?;
            replies.push((
                peer.rank,
                frame.header.status,
                PyArray1::from_vec(py, frame.payload),
            ));
        }
        Ok(replies)
    }

    /// Close every socket; followers see the end of the stream.
    #[pyo3(signature = () -> "None")]
    fn close(&mut self) {
        for peer in self.peers.drain(..) {
            let _ = peer.stream.shutdown(std::net::Shutdown::Both);
        }
        self.listener.close();
        self.closed = true;
    }
}

impl Drop for StepChannelDriver {
    fn drop(&mut self) {
        self.close();
    }
}

/// A follower end: one socket to the driver.
#[pyclass(module = "resin._resin")]
pub struct StepChannelFollower {
    peer: Option<Peer>,
    rank: u16,
}

impl StepChannelFollower {
    fn peer(&mut self) -> PyResult<&mut Peer> {
        self.peer
            .as_mut()
            .ok_or_else(|| PyConnectionError::new_err("the step channel is closed"))
    }
}

#[pymethods]
impl StepChannelFollower {
    /// Connect to the driver at `path` as `rank` and send HELLO with `token`.
    #[new]
    #[pyo3(signature = (path: "str | os.PathLike[str]", token: "bytes", rank: "int", timeout: "float"))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        token: Vec<u8>,
        rank: u16,
        timeout: f64,
    ) -> PyResult<Self> {
        let peer = connect(py, &path, &token, rank, &[], timeout)?;
        Ok(StepChannelFollower {
            peer: Some(peer),
            rank,
        })
    }

    /// The next frame from the driver: (kind, seq, status, payload). `timeout` None waits forever.
    #[pyo3(signature = (timeout: "float | None", what: "str")
        -> "tuple[int, int, int, npt.NDArray[np.uint8]]")]
    fn receive<'py>(
        &mut self,
        py: Python<'py>,
        timeout: Option<f64>,
        what: &str,
    ) -> PyResult<Received<'py>> {
        let until = deadline(timeout)?;
        let peer = self.peer()?;
        let frame = wait_frame(py, peer, until, what)?;
        if frame.header.rank != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "step channel protocol error: a frame from rank {} reached rank {}",
                frame.header.rank, self.rank
            )));
        }
        let Frame { header, payload } = frame;
        Ok((
            header.kind,
            header.seq,
            header.status,
            PyArray1::from_vec(py, payload),
        ))
    }

    /// Send one frame to the driver.
    #[pyo3(signature = (
        kind: "int",
        seq: "int",
        status: "int",
        pieces: "Sequence[npt.NDArray[np.uint8]]",
        timeout: "float",
        what: "str",
    ) -> "None")]
    #[allow(clippy::too_many_arguments)]
    fn send(
        &mut self,
        py: Python<'_>,
        kind: u8,
        seq: u64,
        status: u32,
        pieces: Vec<PyReadonlyArray1<'_, u8>>,
        timeout: f64,
        what: &str,
    ) -> PyResult<()> {
        let header = Header {
            kind,
            rank: self.rank,
            status,
            seq,
        };
        let peer = self.peer()?;
        send_frame(
            py,
            std::slice::from_ref(peer),
            header,
            &pieces,
            timeout,
            what,
        )
    }

    /// Close the socket; the driver sees the end of the stream.
    #[pyo3(signature = () -> "None")]
    fn close(&mut self) {
        if let Some(peer) = self.peer.take() {
            let _ = peer.stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

impl Drop for StepChannelFollower {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peers_exchange_frames_and_see_the_end_of_the_stream() {
        let (a, b) = UnixStream::pair().unwrap();
        let mut left = Peer::new(a, 0, 1).unwrap();
        let right = Peer::new(b, 3, 2).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        let big = vec![9u8; 3 << 20];
        let writer = std::thread::spawn(move || {
            let header = Header {
                kind: 5,
                rank: 3,
                status: 0,
                seq: 11,
            };
            right.send(header, &[b"ab", b"", &big], until).unwrap();
            right
        });
        let frame = loop {
            if let Wait::Frame(frame) = left.wait(Duration::ZERO).unwrap() {
                break frame;
            }
        };
        assert_eq!(
            frame.header,
            Header {
                kind: 5,
                rank: 3,
                status: 0,
                seq: 11
            }
        );
        assert_eq!(frame.payload.len(), 2 + (3 << 20));
        assert_eq!(&frame.payload[..3], b"ab\x09");
        drop(writer.join().unwrap());
        let error = loop {
            match left.wait(Duration::from_micros(50)) {
                Ok(Wait::Slice) => {}
                Ok(Wait::Frame(_)) => panic!("no frame was sent"),
                Err(error) => break error,
            }
        };
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        // Writing to a closed peer is an error, not SIGPIPE.
        let (a, b) = UnixStream::pair().unwrap();
        let peer = Peer::new(a, 0, 1).unwrap();
        drop(b);
        let header = Header {
            kind: 2,
            rank: 0,
            status: 0,
            seq: 1,
        };
        let error = peer.send(header, &[b"x"], until).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn peer_credentials_name_this_process() {
        let (a, _b) = UnixStream::pair().unwrap();
        let (pid, uid) = socket::peer_credentials(&a).unwrap();
        assert_eq!(pid, std::process::id() as i32);
        assert_eq!(uid, socket::effective_uid());
    }
}
