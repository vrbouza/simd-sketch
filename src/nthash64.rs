use std::{array::from_fn, cmp::Ordering};

use packed_seq::{BitSeq, ChunkIt, Delay, PackedSeq, PaddedIt, Seq};
use wide::u32x8;

use crate::nthash_tables;

const LANES: usize = 8;

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

#[derive(Clone, Copy, Debug)]
struct U64Parts {
    lo: u32x8,
    hi: u32x8,
}

impl U64Parts {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            lo: u32x8::ZERO,
            hi: u32x8::ZERO,
        }
    }

    #[inline(always)]
    fn splat(value: u64) -> Self {
        Self {
            lo: u32x8::splat(value as u32),
            hi: u32x8::splat((value >> 32) as u32),
        }
    }

    #[inline(always)]
    fn from_u64s(values: [u64; LANES]) -> Self {
        Self {
            lo: u32x8::new(from_fn(|lane| values[lane] as u32)),
            hi: u32x8::new(from_fn(|lane| (values[lane] >> 32) as u32)),
        }
    }

    #[inline(always)]
    fn to_u64s(self) -> [u64; LANES] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        from_fn(|lane| ((hi[lane] as u64) << 32) | lo[lane] as u64)
    }

    #[inline(always)]
    fn replace_lane(&mut self, lane: usize, value: u64) {
        let mut values = self.to_u64s();
        values[lane] = value;
        *self = Self::from_u64s(values);
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }

    #[inline(always)]
    fn rotl1(self) -> Self {
        Self {
            lo: (self.lo << 1) | (self.hi >> 31),
            hi: (self.hi << 1) | (self.lo >> 31),
        }
    }

    #[inline(always)]
    fn rotr1(self) -> Self {
        Self {
            lo: (self.lo >> 1) | (self.hi << 31),
            hi: (self.hi >> 1) | (self.lo << 31),
        }
    }

    #[inline(always)]
    fn swapbits033(self) -> Self {
        let x = (self.lo ^ (self.hi >> 1)) & u32x8::splat(1);
        Self {
            lo: self.lo ^ x,
            hi: self.hi ^ (x << 1),
        }
    }

    #[inline(always)]
    fn swapbits3263(self) -> Self {
        let x = (self.hi ^ (self.hi >> 31)) & u32x8::splat(1);
        Self {
            lo: self.lo,
            hi: self.hi ^ x ^ (x << 31),
        }
    }

    #[inline(always)]
    #[cfg_attr(not(test), allow(dead_code))]
    fn cmp_eq(self, rhs: Self) -> u32x8 {
        self.lo.cmp_eq(rhs.lo) & self.hi.cmp_eq(rhs.hi)
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> u32x8 {
        let hi_gt = self.hi.cmp_gt(rhs.hi);
        let hi_eq = self.hi.cmp_eq(rhs.hi);
        hi_gt | (hi_eq & self.lo.cmp_gt(rhs.lo))
    }

    #[inline(always)]
    #[cfg_attr(not(test), allow(dead_code))]
    fn cmp_le(self, rhs: Self) -> u32x8 {
        !self.cmp_gt(rhs)
    }

    #[inline(always)]
    fn blend(mask: u32x8, t: Self, f: Self) -> Self {
        Self {
            lo: mask.blend(t.lo, f.lo),
            hi: mask.blend(t.hi, f.hi),
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self::blend(self.cmp_gt(rhs), rhs, self)
    }
}

#[inline(always)]
fn lookup_base_values(bases: u32x8, values: [u64; 4]) -> U64Parts {
    let mut out = U64Parts::splat(values[0]);
    for (base, value) in values.iter().enumerate().skip(1) {
        let mask = bases.cmp_eq(u32x8::splat(base as u32));
        out = U64Parts::blend(mask, U64Parts::splat(*value), out);
    }
    out
}

#[inline(always)]
fn lookup_hash(bases: u32x8) -> U64Parts {
    lookup_base_values(bases, nthash_tables::HASH_LOOKUP)
}

#[inline(always)]
fn lookup_rc_hash(bases: u32x8) -> U64Parts {
    lookup_base_values(bases, nthash_tables::RC_HASH_LOOKUP)
}

#[inline(always)]
fn lookup_remove(bases: u32x8, k: usize) -> U64Parts {
    lookup_base_values(
        bases,
        from_fn(|base| {
            nthash_tables::MS_TAB_31L[(base * 31) + (k % 31)]
                | nthash_tables::MS_TAB_33R[(base * 33) + (k % 33)]
        }),
    )
}

#[inline(always)]
fn rc_bases(bases: u32x8) -> u32x8 {
    bases ^ u32x8::splat(2)
}

#[inline(always)]
fn hash_push_vec(hash: U64Parts, bases: u32x8) -> U64Parts {
    hash.rotl1().swapbits033().xor(lookup_hash(bases))
}

#[inline(always)]
fn roll_hash_vec(hash: U64Parts, old_bases: u32x8, new_bases: u32x8, k: usize) -> U64Parts {
    hash_push_vec(hash, new_bases).xor(lookup_remove(old_bases, k))
}

#[inline(always)]
fn roll_hash_rc_vec(hash: U64Parts, old_bases: u32x8, new_bases: u32x8, k: usize) -> U64Parts {
    hash.xor(lookup_remove(rc_bases(new_bases), k))
        .xor(lookup_rc_hash(old_bases))
        .rotr1()
        .swapbits3263()
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

fn lane_valid_lengths(lane_len: usize, padding: usize) -> [usize; LANES] {
    let total = LANES * lane_len - padding;
    from_fn(|lane| total.saturating_sub(lane * lane_len).min(lane_len))
}

#[derive(Clone, Copy)]
struct LaneState {
    count: usize,
    head: usize,
    primed: bool,
}

impl LaneState {
    fn new() -> Self {
        Self {
            count: 0,
            head: 0,
            primed: false,
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
    seq: PackedSeq<'s>,
    ambiguous: BitSeq<'s>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) {
    if k == 0 {
        return;
    }
    let bases = seq.par_iter_bp_delayed_with_factor(k, Delay(k - 1), 2);
    let ambiguity = ambiguous.par_iter_kmer_ambiguity_aligned(k);
    stream_hashes_from_pairs_ambiguous(bases.zip(ambiguity), k, rc, callback);
}

fn stream_hashes_from_pairs<I>(
    mut pairs: PaddedIt<I>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) where
    I: ChunkIt<(u32x8, u32x8)>,
{
    let mut states: [LaneState; LANES] = from_fn(|_| LaneState::new());
    let mut windows = vec![vec![0_u8; k]; LANES];
    pairs.advance_with(k.saturating_sub(1), |(incoming, _outgoing)| {
        let incoming = incoming.to_array();
        for lane in 0..LANES {
            warmup_base(&mut states[lane], &mut windows[lane], incoming[lane] as u8);
        }
    });

    let lane_len = pairs.it.len();
    let valid_lens = lane_valid_lengths(lane_len, pairs.padding);
    let mut fh = U64Parts::zero();
    let mut rh = U64Parts::zero();

    for (step, (incoming, _outgoing)) in pairs.it.enumerate() {
        if step == 0 {
            fill_initial_windows(&mut states, &mut windows, incoming, step, valid_lens);
            (fh, rh) = init_hashes_ring_vec(&states, &windows, k, rc);
        } else {
            let old_bases = roll_windows(&mut states, &mut windows, incoming, step, valid_lens);
            fh = roll_hash_vec(fh, old_bases, incoming, k);
            if rc {
                rh = roll_hash_rc_vec(rh, old_bases, incoming, k);
            }
        }
        let hash = if rc { fh.min(rh) } else { fh };
        emit_active_hashes(hash, step, valid_lens, callback);
    }
}

fn warmup_base(state: &mut LaneState, window: &mut [u8], base: u8) {
    window[state.count] = base;
    state.count += 1;
}

fn stream_hashes_from_pairs_ambiguous<I>(
    mut pairs: PaddedIt<I>,
    k: usize,
    rc: bool,
    callback: &mut dyn FnMut(u64),
) where
    I: ChunkIt<((u32x8, u32x8), u32x8)>,
{
    let mut states: [LaneState; LANES] = from_fn(|_| LaneState::new());
    let mut windows = vec![vec![0_u8; k]; LANES];
    pairs.advance_with(
        k.saturating_sub(1),
        |((incoming, _outgoing), _ambiguity)| {
            let incoming = incoming.to_array();
            for lane in 0..LANES {
                warmup_base(&mut states[lane], &mut windows[lane], incoming[lane] as u8);
            }
        },
    );

    let lane_len = pairs.it.len();
    let valid_lens = lane_valid_lengths(lane_len, pairs.padding);
    let mut fh = U64Parts::zero();
    let mut rh = U64Parts::zero();

    for (step, ((incoming, _outgoing), ambiguity)) in pairs.it.enumerate() {
        let clean_mask = ambiguity.cmp_eq(u32x8::ZERO);
        let clean = clean_mask.to_array();
        if step == 0 {
            fill_initial_windows(&mut states, &mut windows, incoming, step, valid_lens);
            (fh, rh) = init_hashes_ring_vec(&states, &windows, k, rc);
            for lane in 0..LANES {
                states[lane].primed = step < valid_lens[lane] && clean[lane] != 0;
            }
        } else {
            let primed_before: [bool; LANES] = from_fn(|lane| states[lane].primed);
            let old_bases = roll_windows(&mut states, &mut windows, incoming, step, valid_lens);
            fh = roll_hash_vec(fh, old_bases, incoming, k);
            if rc {
                rh = roll_hash_rc_vec(rh, old_bases, incoming, k);
            }
            for lane in 0..LANES {
                if step >= valid_lens[lane] {
                    continue;
                }
                let is_clean = clean[lane] != 0;
                states[lane].primed = is_clean;
                if is_clean && !primed_before[lane] {
                    let (lane_fh, lane_rh) =
                        init_hashes_ring(&windows[lane], states[lane].head, k, rc);
                    fh.replace_lane(lane, lane_fh);
                    if rc {
                        rh.replace_lane(lane, lane_rh.unwrap_or(0));
                    }
                }
            }
        }
        let hash = if rc { fh.min(rh) } else { fh };
        emit_clean_hashes(hash, step, valid_lens, clean_mask, &states, callback);
    }
}

fn fill_initial_windows(
    states: &mut [LaneState; LANES],
    windows: &mut [Vec<u8>],
    incoming: u32x8,
    step: usize,
    valid_lens: [usize; LANES],
) {
    let incoming = incoming.to_array();
    for lane in 0..LANES {
        if step >= valid_lens[lane] {
            continue;
        }
        let state = &mut states[lane];
        if state.count < windows[lane].len() {
            windows[lane][state.count] = incoming[lane] as u8;
            state.count += 1;
        }
        state.primed = state.count == windows[lane].len();
    }
}

fn roll_windows(
    states: &mut [LaneState; LANES],
    windows: &mut [Vec<u8>],
    incoming: u32x8,
    step: usize,
    valid_lens: [usize; LANES],
) -> u32x8 {
    let incoming = incoming.to_array();
    let mut old_bases = [0; LANES];
    for lane in 0..LANES {
        if step >= valid_lens[lane] {
            continue;
        }
        let state = &mut states[lane];
        old_bases[lane] = windows[lane][state.head] as u32;
        windows[lane][state.head] = incoming[lane] as u8;
        state.head += 1;
        if state.head == windows[lane].len() {
            state.head = 0;
        }
    }
    u32x8::new(old_bases)
}

fn init_hashes_ring_vec(
    states: &[LaneState; LANES],
    windows: &[Vec<u8>],
    k: usize,
    rc: bool,
) -> (U64Parts, U64Parts) {
    let mut fh = U64Parts::zero();
    for offset in 0..k {
        fh = hash_push_vec(fh, ring_bases_vec(states, windows, offset, k));
    }

    let mut rh = U64Parts::zero();
    if rc {
        for offset in (0..k).rev() {
            rh = hash_push_vec(rh, rc_bases(ring_bases_vec(states, windows, offset, k)));
        }
    }
    (fh, rh)
}

fn ring_bases_vec(
    states: &[LaneState; LANES],
    windows: &[Vec<u8>],
    offset: usize,
    k: usize,
) -> u32x8 {
    u32x8::new(from_fn(|lane| {
        windows[lane][(states[lane].head + offset) % k] as u32
    }))
}

fn emit_active_hashes(
    hash: U64Parts,
    step: usize,
    valid_lens: [usize; LANES],
    callback: &mut dyn FnMut(u64),
) {
    let hashes = hash.to_u64s();
    for lane in 0..LANES {
        if step < valid_lens[lane] {
            callback(hashes[lane]);
        }
    }
}

fn emit_clean_hashes(
    hash: U64Parts,
    step: usize,
    valid_lens: [usize; LANES],
    clean_mask: u32x8,
    states: &[LaneState; LANES],
    callback: &mut dyn FnMut(u64),
) {
    let hashes = hash.to_u64s();
    let clean = clean_mask.to_array();
    for lane in 0..LANES {
        if step < valid_lens[lane] && clean[lane] != 0 && states[lane].primed {
            callback(hashes[lane]);
        }
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

    fn mask_to_bools(mask: u32x8) -> [bool; LANES] {
        let mask = mask.to_array();
        from_fn(|lane| mask[lane] != 0)
    }

    #[test]
    fn u64parts_roundtrip() {
        let values = [
            0,
            1,
            u32::MAX as u64,
            (u32::MAX as u64) + 1,
            0x8000_0000_0000_0000,
            0xffff_ffff_0000_0000,
            0x0123_4567_89ab_cdef,
            u64::MAX,
        ];
        assert_eq!(U64Parts::from_u64s(values).to_u64s(), values);
    }

    #[test]
    fn u64parts_xor_and_rotates_match_scalar() {
        let a = [
            0,
            1,
            0x8000_0000,
            0x1_0000_0000,
            0x8000_0000_0000_0000,
            0xffff_ffff_ffff_ffff,
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
        ];
        let b = [
            u64::MAX,
            0x8000_0000_0000_0000,
            0x1_0000_0001,
            0xffff_ffff_0000_0000,
            0x0000_0000_ffff_ffff,
            0x1111_2222_3333_4444,
            0x5555_aaaa_5555_aaaa,
            0,
        ];
        let va = U64Parts::from_u64s(a);
        let vb = U64Parts::from_u64s(b);
        assert_eq!(va.xor(vb).to_u64s(), from_fn(|lane| a[lane] ^ b[lane]));
        assert_eq!(va.rotl1().to_u64s(), from_fn(|lane| a[lane].rotate_left(1)));
        assert_eq!(
            va.rotr1().to_u64s(),
            from_fn(|lane| a[lane].rotate_right(1))
        );
    }

    #[test]
    fn u64parts_swapbits_match_scalar() {
        let values = [
            0,
            1,
            1 << 33,
            (1 << 32) | (1 << 63),
            0xffff_ffff_ffff_ffff,
            0x0123_4567_89ab_cdef,
            0x8000_0001_0000_0001,
            0x0000_0002_0000_0000,
        ];
        let parts = U64Parts::from_u64s(values);
        assert_eq!(
            parts.swapbits033().to_u64s(),
            from_fn(|lane| swapbits033(values[lane]))
        );
        assert_eq!(
            parts.swapbits3263().to_u64s(),
            from_fn(|lane| swapbits3263(values[lane]))
        );
    }

    #[test]
    fn u64parts_min_and_masks_are_lane_wise() {
        let a = [
            0,
            1,
            0x0000_0002_ffff_ffff,
            0x0000_0002_0000_0001,
            0x8000_0000_0000_0000,
            0xffff_ffff_0000_0000,
            42,
            u64::MAX,
        ];
        let b = [
            0,
            2,
            0x0000_0003_0000_0000,
            0x0000_0002_0000_0002,
            0x7fff_ffff_ffff_ffff,
            0xffff_fffe_ffff_ffff,
            42,
            u64::MAX - 1,
        ];
        let va = U64Parts::from_u64s(a);
        let vb = U64Parts::from_u64s(b);
        assert_eq!(va.min(vb).to_u64s(), from_fn(|lane| a[lane].min(b[lane])));
        assert_eq!(
            mask_to_bools(va.cmp_eq(vb)),
            from_fn(|lane| a[lane] == b[lane])
        );
        assert_eq!(
            mask_to_bools(va.cmp_le(vb)),
            from_fn(|lane| a[lane] <= b[lane])
        );
        assert_eq!(
            mask_to_bools(va.cmp_gt(vb)),
            from_fn(|lane| a[lane] > b[lane])
        );
    }

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
    fn simd_matches_scalar_length_edge_cases() {
        for k in [1, 3, 5, 9] {
            for case in [
                b"".as_slice(),
                b"A".as_slice(),
                b"ACG".as_slice(),
                b"ACGTA".as_slice(),
                b"ACGTACGTAC".as_slice(),
            ] {
                let seq = PackedSeqVec::from_ascii(case);
                let bases: Vec<u8> = seq.as_slice().iter_bp().collect();
                let mut scalar = NtHashIterator::new(bases, k, true).collect::<Vec<_>>();
                let mut simd = collect_simd(seq, k, true);
                scalar.sort_unstable();
                simd.sort_unstable();
                assert_eq!(simd, scalar, "k={k} seq={case:?}");
            }
        }
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
