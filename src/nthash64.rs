use std::{array::from_fn, cmp::Ordering};

use packed_seq::{BitSeq, ChunkIt, Delay, PaddedIt, Seq};
use wide::u32x8;

use crate::nthash_tables;

#[inline(always)]
fn rc_base(base: u8) -> u8 {
    base ^ 2
}

#[inline(always)]
fn swapbits033(v: u64) -> u64 {
    let x = (v ^ (v >> 33)) & 1;
    v ^ (x | (x << 33))
}

#[inline(always)]
fn swapbits3263(v: u64) -> u64 {
    let x = ((v >> 32) ^ (v >> 63)) & 1;
    v ^ ((x << 32) | (x << 63))
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub struct NtHashIterator {
    k: usize,
    rc: bool,
    fh: u64,
    rh: Option<u64>,
    index: usize,
    seq: Vec<u8>,
    seq_len: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl NtHashIterator {
    pub fn new(seq: Vec<u8>, k: usize, rc: bool) -> Self {
        let seq_len = seq.len();
        let (fh, rh, index) = if k == 0 || seq_len < k {
            (0, None, seq_len.saturating_add(1))
        } else if let Some((fh, rh, index)) = Self::new_iterator(0, &seq, k, rc) {
            (fh, rh, index)
        } else {
            (0, None, seq_len.saturating_add(1))
        };
        Self {
            k,
            rc,
            fh,
            rh,
            index,
            seq,
            seq_len,
        }
    }

    fn curr_hash(&self) -> u64 {
        if let Some(rev) = self.rh {
            self.fh.min(rev)
        } else {
            self.fh
        }
    }

    fn new_iterator(
        mut start: usize,
        seq: &[u8],
        k: usize,
        rc: bool,
    ) -> Option<(u64, Option<u64>, usize)> {
        let mut fh = 0_u64;
        'outer: while start < seq.len().saturating_sub(k).saturating_add(1) {
            for (i, v) in seq[start..(start + k)].iter().enumerate() {
                if *v > 3 {
                    start += i + 1;
                    fh = 0;
                    continue 'outer;
                }
                fh = hash_push(fh, *v);
            }
            break 'outer;
        }
        if start >= seq.len().saturating_sub(k).saturating_add(1) {
            return None;
        }
        let rh = if rc {
            let mut h = 0_u64;
            for v in seq[start..(start + k)].iter().rev() {
                h = hash_push(h, rc_base(*v));
            }
            Some(h)
        } else {
            None
        };
        Some((fh, rh, start + k))
    }

    fn roll_fwd(&mut self, old_base: u8, new_base: u8) {
        self.fh = roll_hash(self.fh, old_base, new_base, self.k);
        if let Some(rev) = self.rh {
            self.rh = Some(roll_hash_rc(rev, old_base, new_base, self.k));
        }
    }
}

impl Iterator for NtHashIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index.cmp(&self.seq_len) {
            Ordering::Less => {
                let current = self.curr_hash();
                let new_base = self.seq[self.index];
                if new_base > 3 {
                    if let Some((fh, rh, index)) =
                        Self::new_iterator(self.index + 1, &self.seq, self.k, self.rc)
                    {
                        self.fh = fh;
                        self.rh = rh;
                        self.index = index;
                    } else {
                        self.index = self.seq_len.saturating_add(1);
                    }
                } else {
                    self.roll_fwd(self.seq[self.index - self.k], new_base);
                    self.index += 1;
                }
                Some(current)
            }
            Ordering::Equal => {
                self.index += 1;
                Some(self.curr_hash())
            }
            Ordering::Greater => None,
        }
    }
}

#[inline(always)]
fn hash_push(hash: u64, base: u8) -> u64 {
    let hash = hash.rotate_left(1);
    let hash = swapbits033(hash);
    hash ^ nthash_tables::HASH_LOOKUP[base as usize]
}

#[inline(always)]
fn roll_hash(hash: u64, old_base: u8, new_base: u8, k: usize) -> u64 {
    let hash = hash_push(hash, new_base);
    hash ^ (nthash_tables::MS_TAB_31L[(old_base as usize * 31) + (k % 31)]
        | nthash_tables::MS_TAB_33R[(old_base as usize * 33) + (k % 33)])
}

#[inline(always)]
fn roll_hash_rc(hash: u64, old_base: u8, new_base: u8, k: usize) -> u64 {
    let mut h = hash
        ^ (nthash_tables::MS_TAB_31L[(rc_base(new_base) as usize * 31) + (k % 31)]
            | nthash_tables::MS_TAB_33R[(rc_base(new_base) as usize * 33) + (k % 33)]);
    h ^= nthash_tables::RC_HASH_LOOKUP[old_base as usize];
    h = h.rotate_right(1);
    swapbits3263(h)
}

#[inline(always)]
fn init_hashes(window: &[u8], k: usize, rc: bool) -> (u64, Option<u64>) {
    let mut fh = 0;
    for &base in window.iter().take(k) {
        fh = hash_push(fh, base);
    }
    let rh = if rc {
        let mut rh = 0;
        for &base in window.iter().take(k).rev() {
            rh = hash_push(rh, rc_base(base));
        }
        Some(rh)
    } else {
        None
    };
    (fh, rh)
}

fn lane_valid_lengths(lane_len: usize, padding: usize) -> [usize; 8] {
    let total = 8 * lane_len - padding;
    from_fn(|lane| total.saturating_sub(lane * lane_len).min(lane_len))
}

#[derive(Clone, Copy)]
struct LaneState {
    count: usize,
    head: usize,
    primed: bool,
    fh: u64,
    rh: u64,
}

impl LaneState {
    fn new() -> Self {
        Self {
            count: 0,
            head: 0,
            primed: false,
            fh: 0,
            rh: 0,
        }
    }
}

pub fn for_each_hash_simd<'s>(
    seq: impl Seq<'s>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) {
    if k == 0 {
        return;
    }
    let bases = seq.par_iter_bp_delayed(k, Delay(k - 1));
    stream_hashes_from_pairs(bases, k, rc, callback);
}

pub fn for_each_hash_simd_ambiguous<'s>(
    seq: impl Seq<'s>,
    ambiguous: BitSeq<'s>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) {
    if k == 0 {
        return;
    }
    let bases = seq.par_iter_bp_delayed(k, Delay(k - 1));
    let validity = ambiguous
        .iter_kmer_ambiguity(k)
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    stream_hashes_from_pairs_ambiguous(bases, validity, k, rc, callback);
}

fn stream_hashes_from_pairs<I>(
    mut pairs: PaddedIt<I>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) where
    I: ChunkIt<(u32x8, u32x8)>,
{
    let mut states: [LaneState; 8] = from_fn(|_| LaneState::new());
    let mut windows = vec![vec![0_u8; k]; 8];
    pairs.advance_with(k.saturating_sub(1), |(incoming, _outgoing)| {
        let incoming = incoming.to_array();
        for lane in 0..8 {
            warmup_base(&mut states[lane], &mut windows[lane], incoming[lane] as u8);
        }
    });

    let lane_len = pairs.it.len();
    let valid_lens = lane_valid_lengths(lane_len, pairs.padding);

    for (step, (incoming, _outgoing)) in pairs.it.enumerate() {
        let incoming = incoming.to_array();
        for lane in 0..8 {
            if step >= valid_lens[lane] {
                continue;
            }
            process_base(
                &mut states[lane],
                &mut windows[lane],
                incoming[lane] as u8,
                k,
                rc,
                callback,
            );
        }
    }
}

fn warmup_base(state: &mut LaneState, window: &mut [u8], base: u8) {
    window[state.count] = base;
    state.count += 1;
}

fn lane_offsets(valid_lens: [usize; 8]) -> [usize; 8] {
    let mut offsets = [0; 8];
    let mut sum = 0;
    for lane in 0..8 {
        offsets[lane] = sum;
        sum += valid_lens[lane];
    }
    offsets
}

fn stream_hashes_from_pairs_ambiguous<I>(
    mut pairs: PaddedIt<I>,
    validity: Vec<u32>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) where
    I: ChunkIt<(u32x8, u32x8)>,
{
    let mut states: [LaneState; 8] = from_fn(|_| LaneState::new());
    let mut windows = vec![vec![0_u8; k]; 8];
    pairs.advance_with(k.saturating_sub(1), |(incoming, _outgoing)| {
        let incoming = incoming.to_array();
        for lane in 0..8 {
            warmup_base(&mut states[lane], &mut windows[lane], incoming[lane] as u8);
        }
    });

    let lane_len = pairs.it.len();
    let valid_lens = lane_valid_lengths(lane_len, pairs.padding);
    let offsets = lane_offsets(valid_lens);

    for (step, (incoming, _outgoing)) in pairs.it.enumerate() {
        let incoming = incoming.to_array();
        for lane in 0..8 {
            if step >= valid_lens[lane] {
                continue;
            }
            let validity_idx = offsets[lane] + step;
            process_base_masked(
                &mut states[lane],
                &mut windows[lane],
                incoming[lane] as u8,
                validity[validity_idx] == 0,
                k,
                rc,
                callback,
            );
        }
    }
}

fn process_base(
    state: &mut LaneState,
    window: &mut [u8],
    base: u8,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) {
    if state.count < k {
        window[state.count] = base;
        state.count += 1;
        if state.count == k {
            let (fh, rh) = init_hashes_ring(window, 0, k, rc);
            state.fh = fh;
            state.rh = rh.unwrap_or(0);
            state.primed = true;
            callback(rh.map_or(fh, |rev| fh.min(rev)));
        }
        return;
    }

    let old_base = window[state.head];
    window[state.head] = base;
    state.head += 1;
    if state.head == k {
        state.head = 0;
    }
    state.fh = roll_hash(state.fh, old_base, base, k);
    if rc {
        state.rh = roll_hash_rc(state.rh, old_base, base, k);
        callback(state.fh.min(state.rh));
    } else {
        callback(state.fh);
    }
}

fn process_base_masked(
    state: &mut LaneState,
    window: &mut [u8],
    base: u8,
    valid: bool,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) {
    if state.count < k {
        window[state.count] = base;
        state.count += 1;
        if state.count == k {
            if valid {
                let (fh, rh) = init_hashes_ring(window, 0, k, rc);
                state.fh = fh;
                state.rh = rh.unwrap_or(0);
                state.primed = true;
                callback(rh.map_or(fh, |rev| fh.min(rev)));
            } else {
                state.primed = false;
            }
        }
        return;
    }

    let old_base = window[state.head];
    window[state.head] = base;
    state.head += 1;
    if state.head == k {
        state.head = 0;
    }

    if !valid {
        state.primed = false;
        return;
    }

    if !state.primed {
        let (fh, rh) = init_hashes_ring(window, state.head, k, rc);
        state.fh = fh;
        state.rh = rh.unwrap_or(0);
        state.primed = true;
        callback(rh.map_or(fh, |rev| fh.min(rev)));
        return;
    }

    state.fh = roll_hash(state.fh, old_base, base, k);
    if rc {
        state.rh = roll_hash_rc(state.rh, old_base, base, k);
        callback(state.fh.min(state.rh));
    } else {
        callback(state.fh);
    }
}

fn init_hashes_ring(window: &[u8], head: usize, k: usize, rc: bool) -> (u64, Option<u64>) {
    if head == 0 {
        return init_hashes(window, k, rc);
    }
    let mut fh = 0;
    for offset in 0..k {
        let base = window[(head + offset) % k];
        fh = hash_push(fh, base);
    }
    let rh = if rc {
        let mut rh = 0;
        for offset in (0..k).rev() {
            let base = window[(head + offset) % k];
            rh = hash_push(rh, rc_base(base));
        }
        Some(rh)
    } else {
        None
    };
    (fh, rh)
}

#[cfg(test)]
mod test {
    use packed_seq::{PackedNSeqVec, PackedSeqVec, SeqVec};

    use super::*;

    fn collect_simd(seq: impl SeqVec, k: usize, rc: bool) -> Vec<u64> {
        let mut out = vec![];
        for_each_hash_simd(seq.as_slice(), k, rc, &mut |h| out.push(h));
        out
    }

    fn collect_simd_ambiguous(seq: &PackedNSeqVec, k: usize, rc: bool) -> Vec<u64> {
        let mut out = vec![];
        for_each_hash_simd_ambiguous(
            seq.as_slice().seq,
            seq.as_slice().ambiguous,
            k,
            rc,
            &mut |h| out.push(h),
        );
        out
    }

    fn collect_scalar_ambiguous(seq: &PackedNSeqVec, k: usize, rc: bool) -> Vec<u64> {
        let bases: Vec<u8> = seq
            .as_slice()
            .seq
            .iter_bp()
            .zip(seq.as_slice().ambiguous.iter_bp())
            .map(|(base, amb)| if amb == 0 { base } else { 5 })
            .collect();
        NtHashIterator::new(bases, k, rc).collect()
    }

    #[test]
    fn simd_matches_scalar_forward() {
        let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGT");
        let bases: Vec<u8> = seq.as_slice().iter_bp().collect();
        let mut scalar = NtHashIterator::new(bases, 7, false).collect::<Vec<_>>();
        let mut simd = collect_simd(seq, 7, false);
        scalar.sort_unstable();
        simd.sort_unstable();
        assert_eq!(simd, scalar);
    }

    #[test]
    fn simd_matches_scalar_canonical() {
        let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGT");
        let bases: Vec<u8> = seq.as_slice().iter_bp().collect();
        let mut scalar = NtHashIterator::new(bases, 7, true).collect::<Vec<_>>();
        let mut simd = collect_simd(seq, 7, true);
        scalar.sort_unstable();
        simd.sort_unstable();
        assert_eq!(simd, scalar);
    }

    #[test]
    fn simd_matches_scalar_with_ambiguous_bases() {
        let seq = PackedNSeqVec::from_ascii(b"ACGTNNNNACGTACGTNNNNACGT");
        let scalar = collect_scalar_ambiguous(&seq, 5, true);
        let simd = collect_simd_ambiguous(&seq, 5, true);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn simd_matches_scalar_ambiguous_forward_restart_cases() {
        let cases: &[&[u8]] = &[
            b"NACGTACGTACGT",
            b"ACGTACGTACGTN",
            b"ACGTNACGTACGT",
            b"ACGTNNNNACGTACGT",
            b"NNNNACGTACGTNNNN",
            b"ACGTNACGNACGTNACGT",
            b"ACGTNNAC",
            b"NNNNNN",
            b"ACGT",
            b"ACGTN",
        ];

        for &case in cases {
            let seq = PackedNSeqVec::from_ascii(case);
            let mut scalar = collect_scalar_ambiguous(&seq, 5, false);
            let mut simd = collect_simd_ambiguous(&seq, 5, false);
            scalar.sort_unstable();
            simd.sort_unstable();
            assert_eq!(simd, scalar, "seq={:?}", case);
        }
    }

    #[test]
    fn simd_matches_scalar_ambiguous_canonical_restart_cases() {
        let cases: &[&[u8]] = &[
            b"NACGTACGTACGT",
            b"ACGTACGTACGTN",
            b"ACGTNACGTACGT",
            b"ACGTNNNNACGTACGT",
            b"NNNNACGTACGTNNNN",
            b"ACGTNACGNACGTNACGT",
            b"ACGTNNAC",
            b"NNNNNN",
            b"ACGT",
            b"ACGTN",
        ];

        for &case in cases {
            let seq = PackedNSeqVec::from_ascii(case);
            let mut scalar = collect_scalar_ambiguous(&seq, 5, true);
            let mut simd = collect_simd_ambiguous(&seq, 5, true);
            scalar.sort_unstable();
            simd.sort_unstable();
            assert_eq!(simd, scalar, "seq={:?}", case);
        }
    }

    #[test]
    fn simd_matches_scalar_ambiguous_length_edge_cases() {
        for k in [3, 5, 7] {
            for case in [
                b"AN".as_slice(),
                b"ACG".as_slice(),
                b"ACGTN".as_slice(),
                b"NNNNN".as_slice(),
            ] {
                let seq = PackedNSeqVec::from_ascii(case);
                let mut scalar = collect_scalar_ambiguous(&seq, k, true);
                let mut simd = collect_simd_ambiguous(&seq, k, true);
                scalar.sort_unstable();
                simd.sort_unstable();
                assert_eq!(simd, scalar, "k={k} seq={:?}", case);
            }
        }
    }
}
