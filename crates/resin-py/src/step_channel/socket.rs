//! Unix-socket helpers: whole-frame writes without SIGPIPE, peer credentials, timeouts.

use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use super::frame::{Frame, Header, Poll, Reader};

/// How long one blocked read waits before the caller checks its deadline and signals.
pub const SLICE: Duration = Duration::from_millis(100);

/// One end of the channel: a connected stream and its frame reader.
pub struct Peer {
    pub stream: UnixStream,
    pub reader: Reader,
    /// The peer's rank (the driver is 0) and process id, for error messages.
    pub rank: u16,
    pub pid: i32,
}

pub enum Wait {
    Frame(Frame),
    /// No frame arrived within one slice.
    Slice,
}

impl Peer {
    pub fn new(stream: UnixStream, rank: u16, pid: i32) -> io::Result<Peer> {
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(SLICE))?;
        stream.set_write_timeout(Some(SLICE))?;
        Ok(Peer {
            stream,
            reader: Reader::default(),
            rank,
            pid,
        })
    }

    /// The next frame, or `Wait::Slice` if none arrived within one slice. With a nonzero
    /// `spin`, it first polls the socket without blocking for up to that long: a frame that
    /// arrives then does not wait for the scheduler to wake this thread.
    pub fn wait(&mut self, spin: Duration) -> io::Result<Wait> {
        let Peer { stream, reader, .. } = self;
        if !spin.is_zero() {
            let end = Instant::now() + spin;
            loop {
                if let Poll::Frame(frame) = reader.poll(&mut Nonblocking(stream))? {
                    return Ok(Wait::Frame(frame));
                }
                if Instant::now() >= end {
                    break;
                }
                std::hint::spin_loop();
            }
        }
        match reader.poll(&mut ReadStream(stream))? {
            Poll::Frame(frame) => Ok(Wait::Frame(frame)),
            Poll::Idle => Ok(Wait::Slice),
        }
    }

    /// The next frame if one has arrived, without blocking.
    pub fn try_frame(&mut self) -> io::Result<Option<Frame>> {
        let Peer { stream, reader, .. } = self;
        match reader.poll(&mut Nonblocking(stream))? {
            Poll::Frame(frame) => Ok(Some(frame)),
            Poll::Idle => Ok(None),
        }
    }

    /// Whether a frame is already buffered, so that `try_frame` returns it without reading.
    pub fn buffered(&self) -> bool {
        self.reader.buffered()
    }

    /// Write one frame. The write fails with `TimedOut` if it cannot finish by `deadline`.
    pub fn send(&self, header: Header, pieces: &[&[u8]], deadline: Instant) -> io::Result<()> {
        let payload = pieces.iter().map(|piece| piece.len()).sum();
        let lead = header.lead(payload);
        let mut parts: Vec<&[u8]> = Vec::with_capacity(pieces.len() + 1);
        parts.push(&lead);
        parts.extend(pieces.iter().copied().filter(|piece| !piece.is_empty()));
        send_all(self.stream.as_raw_fd(), &parts, deadline)
    }
}

/// How long a wait polls before it blocks: `RESIN_STEP_CHANNEL_SPIN_US` microseconds, 200 by
/// default (0 always blocks). A poll does not wait for the scheduler to wake the thread.
pub fn spin() -> Duration {
    static SPIN: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| {
        let micros = std::env::var("RESIN_STEP_CHANNEL_SPIN_US")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(200);
        Duration::from_micros(micros)
    })
}

struct Nonblocking<'a>(&'a UnixStream);

impl Read for Nonblocking<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        // SAFETY: recv writes at most out.len() bytes into out.
        let count = unsafe {
            libc::recv(
                self.0.as_raw_fd(),
                out.as_mut_ptr() as *mut libc::c_void,
                out.len(),
                libc::MSG_DONTWAIT,
            )
        };
        if count < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(count as usize)
    }
}

struct ReadStream<'a>(&'a mut UnixStream);

impl Read for ReadStream<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.0.read(out)
    }
}

/// sendmsg until every byte is written, with MSG_NOSIGNAL so that a closed peer is an
/// EPIPE error and not a signal.
fn send_all(fd: i32, parts: &[&[u8]], deadline: Instant) -> io::Result<()> {
    let mut parts: Vec<&[u8]> = parts.to_vec();
    let mut first = 0;
    while first < parts.len() {
        let mut vectors: Vec<libc::iovec> = parts[first..]
            .iter()
            .take(512)
            .map(|part| libc::iovec {
                iov_base: part.as_ptr() as *mut _,
                iov_len: part.len(),
            })
            .collect();
        // SAFETY: an all-zero msghdr is valid; it points at iovecs that live until sendmsg returns.
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = vectors.as_mut_ptr();
        message.msg_iovlen = vectors.len() as _;
        // SAFETY: the iovecs point into borrowed slices that outlive the call.
        let written = unsafe { libc::sendmsg(fd, &message, libc::MSG_NOSIGNAL) };
        if written < 0 {
            let error = io::Error::last_os_error();
            match error.kind() {
                io::ErrorKind::Interrupted => continue,
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "the peer did not read the frame before the deadline",
                        ));
                    }
                    continue;
                }
                _ => return Err(error),
            }
        }
        if written == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        let mut written = written as usize;
        while written > 0 {
            let part = parts[first];
            if written >= part.len() {
                written -= part.len();
                first += 1;
            } else {
                parts[first] = &part[written..];
                written = 0;
            }
        }
        while first < parts.len() && parts[first].is_empty() {
            first += 1;
        }
    }
    Ok(())
}

/// The process and user ids of the process at the other end of `stream`.
pub fn peer_credentials(stream: &UnixStream) -> io::Result<(i32, u32)> {
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED writes one ucred into the buffer of the given size.
    let status = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut credentials as *mut libc::ucred as *mut libc::c_void,
            &mut size,
        )
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((credentials.pid, credentials.uid))
}

pub fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() }
}
