//! The words of a host copy of a step's table image, shared by every model's fill: 32-bit writes at byte
//! offsets, and the 256-byte rounding of each table.

/// The size of the smallest table: every table starts at a multiple of it.
pub const SMALL: usize = 256;

/// `nbytes` rounded up to whole SMALL units (at least one).
pub fn round(nbytes: usize) -> usize {
    nbytes.div_ceil(SMALL).max(1) * SMALL
}

/// The 32-bit words of a host copy, written at a table's byte offset and an index in the table.
pub struct Words<'a>(pub &'a mut [i32]);

impl Words<'_> {
    pub fn set(&mut self, offset: usize, index: usize, value: i64) {
        self.0[offset / 4 + index] = value as i32;
    }

    pub fn set_f32(&mut self, offset: usize, index: usize, value: f32) {
        self.0[offset / 4 + index] = value.to_bits() as i32;
    }

    pub fn fill(&mut self, offset: usize, count: usize, value: impl Into<i64>) {
        self.0[offset / 4..offset / 4 + count].fill(value.into() as i32);
    }

    /// Bulk-write `values` (i64, cast to i32) at a table's byte `offset`, word `index`: one bounds
    /// check for the run, then a tight loop the compiler vectorizes (vs per-element `set`).
    pub fn set_many_i64(&mut self, offset: usize, index: usize, values: &[i64]) {
        let start = offset / 4 + index;
        let dst = &mut self.0[start..start + values.len()];
        for (d, &v) in dst.iter_mut().zip(values) {
            *d = v as i32;
        }
    }

    /// Bulk-write `values` (f32, as their bits) at a table's byte `offset`, word `index`.
    pub fn set_many_f32(&mut self, offset: usize, index: usize, values: &[f32]) {
        let start = offset / 4 + index;
        let dst = &mut self.0[start..start + values.len()];
        for (d, &v) in dst.iter_mut().zip(values) {
            *d = v.to_bits() as i32;
        }
    }

    /// Copy `values` (i32 words) verbatim at a table's byte `offset`, word `index` (one memcpy).
    pub fn copy_words(&mut self, offset: usize, index: usize, values: &[i32]) {
        let start = offset / 4 + index;
        self.0[start..start + values.len()].copy_from_slice(values);
    }
}

/// The words of a host copy that holds `nbytes` bytes from its start (4-byte aligned).
pub fn words(host: &mut [u8], nbytes: usize) -> Result<Words<'_>, String> {
    if host.len() < nbytes
        || !(host.as_ptr() as usize).is_multiple_of(4)
        || !nbytes.is_multiple_of(4)
    {
        return Err("the host copy is too small or misaligned".into());
    }
    // SAFETY: the buffer is 4-byte aligned and holds nbytes bytes; every table write stays inside it.
    Ok(Words(unsafe {
        std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<i32>(), nbytes / 4)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_write_at_byte_offsets() {
        let mut host = vec![0u32; 8];
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), 32) };
        let mut w = words(bytes, 32).unwrap();
        w.set(8, 1, -2);
        w.set_f32(0, 0, 1.0);
        w.fill(16, 2, 7);
        assert_eq!(host, [1.0f32.to_bits(), 0, 0, (-2i32) as u32, 7, 7, 0, 0]);
        assert_eq!(
            (round(1), round(256), round(257), round(0)),
            (256, 256, 512, 256)
        );
    }

    #[test]
    fn a_short_or_misaligned_copy_is_refused() {
        let mut host = vec![0u32; 4];
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), 16) };
        assert!(words(bytes, 32).is_err());
        assert!(words(&mut bytes[1..], 8).is_err());
    }
}
