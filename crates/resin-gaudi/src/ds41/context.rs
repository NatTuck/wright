//! DeepSeek-V4.1's Engram context: the tokens before each position that Engram hashes, for every sequence of a
//! runner (the port of `EngramContext` in `tests/ds41_reference.py`).
//!
//! `seen` records the tokens of a sequence at their positions as they are committed, from its first token: tokens
//! at a position it already holds replace those at that position and after. It also records the last `n` tokens
//! of each WINDOW block that they complete, by window block id. A sequence that the Engine starts from cached
//! blocks (a prefix hit, at a block boundary) has no record of its own; `before` then takes its tokens from the
//! record of the block before its first position, which is in its WINDOW table.
//!
//! A sequence's record can end one position past the next step's first position: a plain decode step records its
//! sample at the position after its own (the input of the step that continues it on the device), and when the
//! Engine instead starts the row's next step afresh, that step's first position is the sample's, with the sample
//! as its host token. So a record keeps `n + 1` tokens, and `before` and `seen` go by position, not by the
//! record's end.
//!
//! The Engine does not change a shared block, and the new owner of a free block writes it again before the block
//! can be shared. Thus the record of a block that a plan names is the record of that block's tokens.

use std::collections::HashMap;

use super::PAGE_TOKENS;

/// A sequence's newest recorded tokens: `tokens` at positions `end - tokens.len()` .. `end - 1`.
#[derive(Clone, Debug)]
struct Record {
    end: i64,
    tokens: Vec<i64>,
}

impl Record {
    /// The position of `tokens[0]`.
    fn start(&self) -> i64 {
        self.end - self.tokens.len() as i64
    }
}

#[derive(Clone, Debug)]
pub struct Context {
    n: usize,
    /// sequence id -> its last `n + 1` recorded tokens
    last: HashMap<u64, Record>,
    /// sequence id -> its WINDOW table of the latest plan: first logical block and the window block ids
    windows: HashMap<u64, (i64, Vec<i64>)>,
    /// window block id -> the last `n` tokens of the block (-1: not recorded), `n` a block
    tails: Vec<i64>,
}

impl Context {
    pub fn new(n: usize, window_blocks: usize) -> Result<Self, String> {
        if n < 1 {
            return Err("Engram hashes at least one token before a position".into());
        }
        Ok(Self {
            n,
            last: HashMap::new(),
            windows: HashMap::new(),
            tails: vec![-1; window_blocks * n],
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether the sequence has a record (it is not released).
    pub fn has(&self, sequence: u64) -> bool {
        self.last.contains_key(&sequence)
    }

    /// The tokens at positions `pos0 - n` .. `pos0 - 1` that Engram hashes (those that exist), for a row whose
    /// WINDOW table starts at logical block `first` with window block ids `ids`. They come from the sequence's
    /// record when it holds them, else from the record of the block before `pos0`.
    pub fn before(
        &mut self,
        sequence: u64,
        pos0: i64,
        first: i64,
        ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        self.windows.insert(sequence, (first, ids.to_vec()));
        if pos0 == 0 {
            return Ok(Vec::new());
        }
        let from = pos0 - self.n.min(pos0 as usize) as i64;
        if let Some(record) = self.last.get(&sequence)
            && record.start() <= from
            && pos0 <= record.end
        {
            let at = (from - record.start()) as usize;
            return Ok(record.tokens[at..at + (pos0 - from) as usize].to_vec());
        }
        self.restore(sequence, pos0, first, ids)?;
        Ok(self.last[&sequence].tokens.clone())
    }

    fn restore(&mut self, sequence: u64, pos0: i64, first: i64, ids: &[i64]) -> Result<(), String> {
        let block = pos0 / PAGE_TOKENS - 1 - first;
        if pos0 % PAGE_TOKENS != 0 || block < 0 || block as usize >= ids.len() {
            return Err(format!(
                "sequence {sequence} starts at position {pos0}, which is not after a block in its WINDOW table"
            ));
        }
        let tail = self.tail(ids[block as usize])?;
        if tail.iter().any(|&token| token < 0) {
            return Err(format!(
                "sequence {sequence} starts at position {pos0} after a block with no recorded tokens"
            ));
        }
        let keep = self.n.min(pos0 as usize);
        let tokens = tail[self.n - keep..].to_vec();
        self.last.insert(sequence, Record { end: pos0, tokens });
        Ok(())
    }

    fn tail(&self, id: i64) -> Result<&[i64], String> {
        let at = usize::try_from(id)
            .ok()
            .filter(|&id| (id + 1) * self.n <= self.tails.len())
            .ok_or_else(|| format!("window block {id} is outside the context's blocks"))?;
        Ok(&self.tails[at * self.n..(at + 1) * self.n])
    }

    /// Record `tokens`, committed at positions `pos0` and after, in place of any tokens the record holds at
    /// `pos0` and after. Call after `before` for a row of this sequence whose WINDOW table covers those positions.
    pub fn seen(&mut self, sequence: u64, pos0: i64, tokens: &[i64]) -> Result<(), String> {
        // the recorded tokens before pos0, when the record reaches pos0
        let mut full = match self.last.get(&sequence) {
            Some(record) if pos0 > 0 && record.start() <= pos0 && pos0 <= record.end => {
                record.tokens[..(pos0 - record.start()) as usize].to_vec()
            }
            _ => Vec::new(),
        };
        let start = pos0 - full.len() as i64; // the position of full[0]
        full.extend_from_slice(tokens);
        let n = self.n as i64;
        let end = start + full.len() as i64;
        if let Some((first, ids)) = self.windows.get(&sequence) {
            let mut e = (pos0 + 1 + PAGE_TOKENS - 1).div_euclid(PAGE_TOKENS) * PAGE_TOKENS;
            while e <= end {
                let block = e / PAGE_TOKENS - 1 - first;
                if e - n >= start && block >= 0 && (block as usize) < ids.len() {
                    let id = ids[block as usize];
                    let at = usize::try_from(id)
                        .ok()
                        .filter(|&id| (id + 1) * self.n <= self.tails.len())
                        .ok_or_else(|| {
                            format!("window block {id} is outside the context's blocks")
                        })?;
                    let from = (e - n - start) as usize;
                    self.tails[at * self.n..(at + 1) * self.n]
                        .copy_from_slice(&full[from..from + self.n]);
                }
                e += PAGE_TOKENS;
            }
        }
        let keep = (self.n + 1).min(full.len());
        full.drain(..full.len() - keep);
        self.last.insert(sequence, Record { end, tokens: full });
        Ok(())
    }

    /// Forget released sequences. The records of their blocks stay: cached blocks outlive their sequence.
    pub fn release(&mut self, sequences: &[u64]) {
        for sequence in sequences {
            self.last.remove(sequence);
            self.windows.remove(sequence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_hit_restores_the_block_s_tail() {
        let mut context = Context::new(3, 4).unwrap();
        // sequence 1 fills blocks 2 and 0 (logical 0 and 1)
        assert_eq!(context.before(1, 0, 0, &[2, 0]).unwrap(), Vec::<i64>::new());
        let tokens: Vec<i64> = (0..200).collect();
        context.seen(1, 0, &tokens).unwrap();
        assert_eq!(
            context.before(1, 200, 0, &[2, 0]).unwrap(),
            vec![197, 198, 199]
        );
        // sequence 2 starts at 128 from block 2's record
        assert_eq!(
            context.before(2, 128, 0, &[2, 3]).unwrap(),
            vec![125, 126, 127]
        );
        // a block without a record, or a start inside a block, is an error
        assert!(context.before(3, 128, 0, &[3]).is_err());
        assert!(context.before(4, 100, 0, &[2]).is_err());
        context.release(&[1]);
        assert!(!context.has(1) && context.has(2));
    }

    #[test]
    fn a_record_that_holds_pos0_is_read_and_replaced_by_position() {
        let mut context = Context::new(3, 4).unwrap();
        let ids = [2, 0];
        context.before(1, 0, 0, &ids).unwrap();
        let tokens: Vec<i64> = (0..100).collect();
        context.seen(1, 0, &tokens).unwrap();
        // a plain decode step at 100 with its host token, then its sample recorded at 101
        assert_eq!(context.before(1, 100, 0, &ids).unwrap(), vec![97, 98, 99]);
        context.seen(1, 100, &[100]).unwrap();
        context.seen(1, 101, &[101]).unwrap();
        // the next step starts afresh at 101 with the sample as its host token: the record ends at 102
        assert_eq!(context.before(1, 101, 0, &ids).unwrap(), vec![98, 99, 100]);
        context.seen(1, 101, &[101]).unwrap();
        assert_eq!(context.before(1, 102, 0, &ids).unwrap(), vec![99, 100, 101]);
        // tokens at a recorded position replace those at it and after
        context.seen(1, 101, &[7]).unwrap();
        assert_eq!(context.before(1, 102, 0, &ids).unwrap(), vec![99, 100, 7]);
        context.seen(1, 100, &[5, 6, 8]).unwrap();
        assert_eq!(context.before(1, 102, 0, &ids).unwrap(), vec![99, 5, 6]);
        assert_eq!(context.before(1, 103, 0, &ids).unwrap(), vec![5, 6, 8]);
        // a speculative step from an anchor whose position the record already holds commits [anchor, drafts]
        context.seen(1, 103, &[103]).unwrap();
        assert_eq!(context.before(1, 103, 0, &ids).unwrap(), vec![5, 6, 8]);
        context.seen(1, 103, &[103, 104, 105]).unwrap();
        assert_eq!(
            context.before(1, 106, 0, &ids).unwrap(),
            vec![103, 104, 105]
        );
    }

    #[test]
    fn a_fresh_step_at_the_record_s_last_position_keeps_block_tails() {
        // the next step starts afresh at the last position of block 0 (127) and at the one before it (126)
        for fresh in [126, 127] {
            let mut context = Context::new(3, 4).unwrap();
            let ids = [2, 0];
            context.before(1, 0, 0, &ids).unwrap();
            let tokens: Vec<i64> = (0..fresh - 1).collect();
            context.seen(1, 0, &tokens).unwrap();
            // a plain decode step at fresh - 1; its sample is recorded at fresh
            context.before(1, fresh - 1, 0, &ids).unwrap();
            context.seen(1, fresh - 1, &[fresh - 1]).unwrap();
            context.seen(1, fresh, &[fresh]).unwrap();
            for p in fresh..130 {
                // a fresh step at p, then its sample at p + 1
                let want: Vec<i64> = (p - 3..p).collect();
                assert_eq!(context.before(1, p, 0, &ids).unwrap(), want);
                context.seen(1, p, &[p]).unwrap();
                context.seen(1, p + 1, &[p + 1]).unwrap();
            }
            // sequence 2 starts at 128 from block 2's record
            assert_eq!(
                context.before(2, 128, 0, &[2, 3]).unwrap(),
                vec![125, 126, 127]
            );
        }
    }
}
