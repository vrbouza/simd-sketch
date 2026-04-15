use std::cmp::Ordering;

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
                fh = fh.rotate_left(1);
                fh = swapbits033(fh);
                fh ^= nthash_tables::HASH_LOOKUP[*v as usize];
            }
            break 'outer;
        }
        if start >= seq.len().saturating_sub(k).saturating_add(1) {
            return None;
        }
        let rh = if rc {
            let mut h = 0_u64;
            for v in seq[start..(start + k)].iter().rev() {
                h = h.rotate_left(1);
                h = swapbits033(h);
                h ^= nthash_tables::RC_HASH_LOOKUP[*v as usize];
            }
            Some(h)
        } else {
            None
        };
        Some((fh, rh, start + k))
    }

    fn roll_fwd(&mut self, old_base: u8, new_base: u8) {
        self.fh = self.fh.rotate_left(1);
        self.fh = swapbits033(self.fh);
        self.fh ^= nthash_tables::HASH_LOOKUP[new_base as usize];
        self.fh ^= nthash_tables::MS_TAB_31L[(old_base as usize * 31) + (self.k % 31)]
            | nthash_tables::MS_TAB_33R[(old_base as usize * 33) + (self.k % 33)];

        if let Some(rev) = self.rh {
            let mut h = rev
                ^ (nthash_tables::MS_TAB_31L[(rc_base(new_base) as usize * 31) + (self.k % 31)]
                    | nthash_tables::MS_TAB_33R[(rc_base(new_base) as usize * 33) + (self.k % 33)]);
            h ^= nthash_tables::RC_HASH_LOOKUP[old_base as usize];
            h = h.rotate_right(1);
            h = swapbits3263(h);
            self.rh = Some(h);
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
