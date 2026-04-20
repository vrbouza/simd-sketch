#![allow(dead_code)]

use packed_seq::{L, Simd as S};

/// Append subset of values indicated by `mask` to a vector.
#[inline(always)]
pub unsafe fn append_from_mask<T: From<u32>>(vals: S, mask: S, v: &mut [T], write_idx: &mut usize) {
    let vals = vals.to_array();
    let mask = mask.to_array();
    for lane in 0..L {
        if mask[lane] > 0 {
            unsafe {
                v.as_mut_ptr().add(*write_idx).write(vals[lane].into());
            }
            *write_idx += 1;
        }
    }
}
