//! Frames of the step channel.
//!
//! A frame is a u64 LE length, then a 16-byte header, then the payload. The
//! length counts the header and the payload. The header is:
//!
//! ```text
//! byte 0      kind
//! byte 1      zero
//! bytes 2..4  rank of the sender, u16 LE (0 for the driver)
//! bytes 4..8  status, u32 LE (0 = ok)
//! bytes 8..16 sequence number, u64 LE
//! ```
//!
//! The channel does not interpret kinds, statuses or payloads.

use std::io::{self, Read};

pub const PREFIX: usize = 8;
pub const HEADER: usize = 16;
pub const LEAD: usize = PREFIX + HEADER;
/// The largest frame either side accepts. A larger length is a protocol error.
pub const MAX_FRAME: u64 = 1 << 34;
/// Payloads up to this size are copied from the read buffer; larger ones are read into their own storage.
const BUFFER: usize = 64 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub kind: u8,
    pub rank: u16,
    pub status: u32,
    pub seq: u64,
}

impl Header {
    /// The length prefix and the header of a frame with `payload` bytes.
    pub fn lead(&self, payload: usize) -> [u8; LEAD] {
        let mut lead = [0u8; LEAD];
        lead[..PREFIX].copy_from_slice(&((HEADER + payload) as u64).to_le_bytes());
        lead[PREFIX] = self.kind;
        lead[PREFIX + 2..PREFIX + 4].copy_from_slice(&self.rank.to_le_bytes());
        lead[PREFIX + 4..PREFIX + 8].copy_from_slice(&self.status.to_le_bytes());
        lead[PREFIX + 8..LEAD].copy_from_slice(&self.seq.to_le_bytes());
        lead
    }

    fn parse(lead: &[u8]) -> io::Result<(Header, usize)> {
        let length = u64::from_le_bytes(lead[..PREFIX].try_into().unwrap());
        if !(HEADER as u64..=MAX_FRAME).contains(&length) {
            return Err(invalid(format!("frame length {length} is out of range")));
        }
        let header = Header {
            kind: lead[PREFIX],
            rank: u16::from_le_bytes(lead[PREFIX + 2..PREFIX + 4].try_into().unwrap()),
            status: u32::from_le_bytes(lead[PREFIX + 4..PREFIX + 8].try_into().unwrap()),
            seq: u64::from_le_bytes(lead[PREFIX + 8..LEAD].try_into().unwrap()),
        };
        Ok((header, length as usize - HEADER))
    }
}

#[derive(Debug)]
pub struct Frame {
    pub header: Header,
    pub payload: Vec<u8>,
}

pub fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Splits a byte stream into frames. A frame that is not complete stays in the reader
/// between calls, so a read can stop at a timeout and continue later.
pub struct Reader {
    buffer: Box<[u8]>,
    start: usize,
    end: usize,
    partial: Option<(Header, Vec<u8>, usize)>,
}

impl Default for Reader {
    fn default() -> Self {
        Reader {
            buffer: vec![0u8; BUFFER].into_boxed_slice(),
            start: 0,
            end: 0,
            partial: None,
        }
    }
}

pub enum Poll {
    Frame(Frame),
    /// The source had no data before its read timeout.
    Idle,
}

impl Reader {
    /// The next frame. Each read of `source` that times out returns `Poll::Idle`.
    pub fn poll(&mut self, source: &mut impl Read) -> io::Result<Poll> {
        loop {
            if let Some(frame) = self.take()? {
                return Ok(Poll::Frame(frame));
            }
            let result = match &mut self.partial {
                // A large payload reads straight into its own storage.
                Some((_, payload, filled)) if payload.len() - *filled > BUFFER / 2 => source
                    .read(&mut payload[*filled..])
                    .inspect(|&n| *filled += n),
                _ => {
                    self.compact();
                    let space = &mut self.buffer[self.end..];
                    source.read(space).inspect(|&n| self.end += n)
                }
            };
            match result {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the peer closed the step channel",
                    ));
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(Poll::Idle);
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Whether the buffer holds a whole frame's lead (a frame may be ready without a read).
    pub fn buffered(&self) -> bool {
        self.partial.is_none() && self.end - self.start >= LEAD
    }

    fn compact(&mut self) {
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        } else if self.buffer.len() - self.end < LEAD {
            self.buffer.copy_within(self.start..self.end, 0);
            self.end -= self.start;
            self.start = 0;
        }
    }

    fn take(&mut self) -> io::Result<Option<Frame>> {
        if self.partial.is_none() {
            if self.end - self.start < LEAD {
                return Ok(None);
            }
            let (header, size) = Header::parse(&self.buffer[self.start..self.start + LEAD])?;
            self.start += LEAD;
            self.partial = Some((header, vec![0u8; size], 0));
        }
        let (_, payload, filled) = self.partial.as_mut().unwrap();
        let count = (payload.len() - *filled).min(self.end - self.start);
        payload[*filled..*filled + count]
            .copy_from_slice(&self.buffer[self.start..self.start + count]);
        *filled += count;
        self.start += count;
        if *filled < payload.len() {
            return Ok(None);
        }
        let (header, payload, _) = self.partial.take().unwrap();
        Ok(Some(Frame { header, payload }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that hands out its bytes in fixed pieces, then times out once, then repeats.
    struct Pieces {
        data: Vec<u8>,
        at: usize,
        piece: usize,
        idle: bool,
    }

    impl Read for Pieces {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.idle {
                self.idle = false;
                return Err(io::ErrorKind::WouldBlock.into());
            }
            self.idle = true;
            let count = self.piece.min(out.len()).min(self.data.len() - self.at);
            out[..count].copy_from_slice(&self.data[self.at..self.at + count]);
            self.at += count;
            Ok(count)
        }
    }

    fn frame(header: Header, payload: &[u8]) -> Vec<u8> {
        let mut bytes = header.lead(payload.len()).to_vec();
        bytes.extend_from_slice(payload);
        bytes
    }

    fn read_all(reader: &mut Reader, source: &mut Pieces, count: usize) -> Vec<Frame> {
        let mut frames = Vec::new();
        while frames.len() < count {
            if let Poll::Frame(frame) = reader.poll(source).unwrap() {
                frames.push(frame);
            }
        }
        frames
    }

    #[test]
    fn frames_survive_any_split_and_timeouts() {
        let big: Vec<u8> = (0..200_000u32).map(|i| (i * 7) as u8).collect();
        let headers = [
            Header {
                kind: 2,
                rank: 0,
                status: 0,
                seq: 1,
            },
            Header {
                kind: 3,
                rank: 7,
                status: 1,
                seq: u64::MAX,
            },
            Header {
                kind: 5,
                rank: 300,
                status: 0,
                seq: 2,
            },
        ];
        let payloads: [&[u8]; 3] = [b"", &big, b"error text"];
        let mut data = Vec::new();
        for (header, payload) in headers.iter().zip(payloads) {
            data.extend(frame(*header, payload));
        }
        for piece in [1, 5, 23, 24, 25, 4096, 1 << 20] {
            let mut source = Pieces {
                data: data.clone(),
                at: 0,
                piece,
                idle: false,
            };
            let frames = read_all(&mut Reader::default(), &mut source, 3);
            for ((frame, header), payload) in frames.iter().zip(headers).zip(payloads) {
                assert_eq!(frame.header, header);
                assert_eq!(frame.payload, payload);
            }
        }
    }

    #[test]
    fn end_of_stream_and_bad_lengths_are_errors() {
        let mut source = Pieces {
            data: vec![1, 2, 3],
            at: 0,
            piece: 8,
            idle: false,
        };
        let mut reader = Reader::default();
        let error = loop {
            match reader.poll(&mut source) {
                Ok(_) => {}
                Err(error) => break error,
            }
        };
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        for length in [0u64, HEADER as u64 - 1, MAX_FRAME + 1] {
            let mut data = length.to_le_bytes().to_vec();
            data.extend([0u8; HEADER]);
            let mut source = Pieces {
                data,
                at: 0,
                piece: 64,
                idle: false,
            };
            let mut reader = Reader::default();
            let error = loop {
                match reader.poll(&mut source) {
                    Ok(_) => {}
                    Err(error) => break error,
                }
            };
            assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{length}");
        }
    }
}
