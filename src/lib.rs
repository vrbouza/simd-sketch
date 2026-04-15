pub mod classify;
mod intrinsics;
mod nthash64;
mod nthash_tables;

use std::{
    collections::{HashMap, hash_map::Entry},
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use itertools::Itertools;
use packed_seq::{PackedNSeq, Seq};
use seq_hash::KmerHasher;

type FwdNtHasher = seq_hash::NtHasher<false, 1>;
type RcNtHasher = seq_hash::NtHasher<true, 1>;

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
pub enum HashMode {
    Legacy32,
    NtHash64,
}

#[derive(bincode::Encode, bincode::Decode, Debug)]
pub enum Sketch {
    BottomSketch(BottomSketch),
    BucketSketch(BucketSketch),
}

#[derive(bincode::Encode, bincode::Decode, Debug)]
pub enum BitSketch {
    B64(Vec<u64>),
    B32(Vec<u32>),
    B16(Vec<u16>),
    B8(Vec<u8>),
    B1(Vec<u64>),
}

impl BitSketch {
    fn new(b: usize, vals: &[u64]) -> Self {
        match b {
            64 => BitSketch::B64(vals.to_vec()),
            32 => BitSketch::B32(vals.iter().map(|x| *x as u32).collect()),
            16 => BitSketch::B16(vals.iter().map(|x| *x as u16).collect()),
            8 => BitSketch::B8(vals.iter().map(|x| *x as u8).collect()),
            1 => BitSketch::B1({
                assert_eq!(vals.len() % 64, 0);
                vals.chunks_exact(64)
                    .map(|xs| {
                        xs.iter()
                            .enumerate()
                            .fold(0u64, |bits, (i, x)| bits | (((x & 1) as u64) << i))
                    })
                    .collect()
            }),
            _ => panic!("Unsupported bit width. Must be 1, 8, 16, 32, or 64."),
        }
    }

    fn len(&self) -> usize {
        match self {
            BitSketch::B64(v) => v.len(),
            BitSketch::B32(v) => v.len(),
            BitSketch::B16(v) => v.len(),
            BitSketch::B8(v) => v.len(),
            BitSketch::B1(v) => 64 * v.len(),
        }
    }
}

#[derive(bincode::Encode, bincode::Decode, Debug)]
pub struct BottomSketch {
    pub hash_mode: HashMode,
    pub rc: bool,
    pub k: usize,
    pub seed: u32,
    pub count: usize,
    pub bottom: Vec<u64>,
}

#[derive(bincode::Encode, bincode::Decode, Debug)]
pub struct BucketSketch {
    pub hash_mode: HashMode,
    pub rc: bool,
    pub k: usize,
    pub b: usize,
    pub seed: u32,
    pub count: usize,
    pub buckets: BitSketch,
    pub empty: Vec<u64>,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum SketchAlg {
    Bottom,
    Bottom2,
    Bottom3,
    Bucket,
}

#[derive(clap::Args, Copy, Clone, Debug, Eq, PartialEq)]
pub struct SketchParams {
    #[arg(long, default_value_t = SketchAlg::Bucket)]
    #[arg(value_enum)]
    pub alg: SketchAlg,
    #[arg(long, default_value_t = HashMode::NtHash64)]
    #[arg(value_enum)]
    pub hash_mode: HashMode,
    #[arg(
        long = "fwd",
        num_args(0),
        action = clap::builder::ArgAction::Set,
        default_value_t = true,
        default_missing_value = "false",
    )]
    pub rc: bool,
    #[arg(short, default_value_t = 31)]
    pub k: usize,
    #[arg(short, default_value_t = 10000)]
    pub s: usize,
    #[arg(short, default_value_t = 8)]
    pub b: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u32,
    #[arg(long, default_value_t = 0)]
    pub count: usize,
    #[arg(short, long, default_value_t = 1)]
    pub coverage: usize,
    #[arg(skip = true)]
    pub filter_empty: bool,
    #[arg(long)]
    pub filter_out_n: bool,
}

pub struct Sketcher {
    params: SketchParams,
    rc_hasher: RcNtHasher,
    fwd_hasher: FwdNtHasher,
    factor: AtomicU64,
}

fn compute_mash_distance(j: f32, k: usize) -> f32 {
    assert!(j >= 0.0, "Jaccard similarity {j} should not be negative");
    let mash_dist = -(2. * j / (1. + j)).ln() / k as f32;
    mash_dist.max(0.0)
}

impl Sketch {
    pub fn to_params(&self) -> SketchParams {
        match self {
            Sketch::BottomSketch(sketch) => SketchParams {
                alg: SketchAlg::Bottom,
                hash_mode: sketch.hash_mode,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.bottom.len(),
                b: 0,
                seed: sketch.seed,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false,
            },
            Sketch::BucketSketch(sketch) => SketchParams {
                alg: SketchAlg::Bucket,
                hash_mode: sketch.hash_mode,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.buckets.len(),
                b: sketch.b,
                seed: sketch.seed,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false,
            },
        }
    }

    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        match (self, other) {
            (Sketch::BottomSketch(a), Sketch::BottomSketch(b)) => a.jaccard_similarity(b),
            (Sketch::BucketSketch(a), Sketch::BucketSketch(b)) => a.jaccard_similarity(b),
            _ => panic!("Sketches are of different types"),
        }
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        let k = match self {
            Sketch::BottomSketch(sketch) => sketch.k,
            Sketch::BucketSketch(sketch) => sketch.k,
        };
        compute_mash_distance(j, k)
    }
}

impl BottomSketch {
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.hash_mode, other.hash_mode);
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        let a = &self.bottom;
        let b = &other.bottom;
        assert_eq!(a.len(), b.len());
        let mut intersection_size = 0;
        let mut union_size = 0;
        let mut i = 0;
        let mut j = 0;
        while union_size < a.len() {
            intersection_size += (a[i] == b[j]) as usize;
            let di = (a[i] <= b[j]) as usize;
            let dj = (a[i] >= b[j]) as usize;
            i += di;
            j += dj;
            union_size += 1;
        }
        intersection_size as f32 / a.len() as f32
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        compute_mash_distance(self.jaccard_similarity(other), self.k)
    }
}

impl BucketSketch {
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.hash_mode, other.hash_mode);
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        assert_eq!(self.b, other.b);
        let both_empty = self.both_empty(other);
        match (&self.buckets, &other.buckets) {
            (BitSketch::B64(a), BitSketch::B64(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B32(a), BitSketch::B32(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B16(a), BitSketch::B16(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B8(a), BitSketch::B8(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B1(a), BitSketch::B1(b)) => Self::b1_similarity(a, b, both_empty),
            _ => panic!("Bit width mismatch"),
        }
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        compute_mash_distance(self.jaccard_similarity(other), self.k)
    }

    fn inner_similarity<T: Eq>(a: &[T], b: &[T], both_empty: usize) -> f32 {
        assert_eq!(a.len(), b.len());
        if a.len() == both_empty {
            return 0.0;
        }
        let f = 1.0
            - std::iter::zip(a, b)
                .map(|(a, b)| (a != b) as u32)
                .sum::<u32>() as f32
                / (a.len() - both_empty) as f32;
        let bits = (size_of::<T>() * 8) as i32;
        let bb = 2f64.powi(bits);
        (((bb * f as f64) - 1.0).max(0.0) / (bb - 1.0)) as f32
    }

    fn b1_similarity(a: &[u64], b: &[u64], both_empty: usize) -> f32 {
        assert_eq!(a.len(), b.len());
        let denom = 64 * a.len() - both_empty;
        if denom == 0 {
            return 0.0;
        }
        let f = 1.0
            - std::iter::zip(a, b)
                .map(|(a, b)| (*a ^ *b).count_ones())
                .sum::<u32>() as f32
                / denom as f32;
        (2. * f - 1.).max(0.0)
    }

    fn both_empty(&self, other: &Self) -> usize {
        std::iter::zip(&self.empty, &other.empty)
            .map(|(a, b)| (a & b).count_ones())
            .sum::<u32>() as usize
    }
}

impl SketchParams {
    pub fn build(&self) -> Sketcher {
        let mut params = *self;
        if params.hash_mode == HashMode::NtHash64 && params.seed != 0 {
            panic!("NtHash64 does not support non-zero seeds");
        }
        let factor = match params.alg {
            SketchAlg::Bottom | SketchAlg::Bottom2 | SketchAlg::Bottom3 => {
                params.b = 0;
                13
            }
            SketchAlg::Bucket => params.s.max(2).ilog2() as u64 * 10,
        };
        Sketcher {
            params,
            rc_hasher: RcNtHasher::new_with_seed(params.k, params.seed),
            fwd_hasher: FwdNtHasher::new_with_seed(params.k, params.seed),
            factor: AtomicU64::new(factor.max(10)),
        }
    }

    pub fn default(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            hash_mode: HashMode::NtHash64,
            rc: true,
            k,
            s: 32768,
            b: 1,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: true,
            filter_out_n: false,
        }
    }

    pub fn default_fast_sketching(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            hash_mode: HashMode::NtHash64,
            rc: true,
            k,
            s: 8192,
            b: 8,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: false,
        }
    }
}

impl Sketcher {
    pub fn params(&self) -> &SketchParams {
        &self.params
    }

    pub fn sketch(&self, seq: impl Sketchable) -> Sketch {
        self.sketch_seqs(&[seq])
    }

    pub fn sketch_seqs(&self, seqs: &[impl Sketchable]) -> Sketch {
        match self.params.alg {
            SketchAlg::Bottom | SketchAlg::Bottom2 | SketchAlg::Bottom3 => {
                Sketch::BottomSketch(self.bottom_sketch(seqs))
            }
            SketchAlg::Bucket => Sketch::BucketSketch(self.bucket_sketch(seqs)),
        }
    }

    fn max_hash(&self) -> u64 {
        match self.params.hash_mode {
            HashMode::Legacy32 => u32::MAX as u64,
            HashMode::NtHash64 => u64::MAX,
        }
    }

    fn num_kmers(&self, seqs: &[impl Sketchable]) -> usize {
        seqs.iter()
            .map(|seq| seq.len().saturating_sub(self.params.k).saturating_add(1))
            .sum()
    }

    fn bottom_sketch(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        let n = self.num_kmers(seqs);
        let max_hash = self.max_hash();
        let mut out = vec![];
        if n == 0 {
            return BottomSketch {
                hash_mode: self.params.hash_mode,
                rc: self.params.rc,
                k: self.params.k,
                seed: self.params.seed,
                count: self.params.count,
                bottom: vec![max_hash; self.params.s],
            };
        }
        loop {
            let target = (max_hash as u128).saturating_mul(self.params.s as u128)
                / (n / self.params.coverage.max(1)).max(1) as u128;
            let factor = self.factor.load(Relaxed);
            let bound = (target.saturating_mul(factor as u128) / 10).min(max_hash as u128) as u64;
            self.collect_up_to_bound(seqs, bound, &mut out, usize::MAX, |_| 0);
            if bound == max_hash || out.len() >= self.params.s {
                out.sort_unstable();
                out.dedup();
                if bound == max_hash || out.len() >= self.params.s {
                    out.resize(self.params.s, max_hash);
                    return BottomSketch {
                        hash_mode: self.params.hash_mode,
                        rc: self.params.rc,
                        k: self.params.k,
                        seed: self.params.seed,
                        count: self.params.count,
                        bottom: out,
                    };
                }
            }
            let new_factor = factor + factor.div_ceil(4);
            self.factor.fetch_max(new_factor, Relaxed);
        }
    }

    fn bucket_sketch(&self, seqs: &[impl Sketchable]) -> BucketSketch {
        let n = self.num_kmers(seqs);
        let max_hash = self.max_hash();
        let mut out = vec![];
        let mut buckets = vec![max_hash; self.params.s];
        if n == 0 {
            let empty = if self.params.filter_empty {
                vec![u64::MAX; self.params.s.div_ceil(64)]
            } else {
                vec![]
            };
            let reduced = vec![max_hash; self.params.s];
            return BucketSketch {
                hash_mode: self.params.hash_mode,
                rc: self.params.rc,
                k: self.params.k,
                b: if self.params.hash_mode == HashMode::NtHash64 {
                    self.params.b
                } else {
                    self.params.b.min(32)
                },
                seed: self.params.seed,
                count: self.params.count,
                buckets: BitSketch::new(self.effective_b(), &reduced),
                empty,
            };
        }
        loop {
            buckets.fill(max_hash);
            let target = (max_hash as u128).saturating_mul(self.params.s as u128)
                / (n / self.params.coverage.max(1)).max(1) as u128;
            let factor = self.factor.load(Relaxed);
            let bound = (target.saturating_mul(factor as u128) / 10).min(max_hash as u128) as u64;
            self.collect_up_to_bound(seqs, bound, &mut out, usize::MAX, |_| 0);
            let mut seen = HashMap::with_capacity(4 * self.params.s.max(1));
            for &hash in &out {
                let bucket = (hash % self.params.s as u64) as usize;
                let min = &mut buckets[bucket];
                if self.params.count <= 1 {
                    *min = (*min).min(hash);
                    continue;
                }
                if hash > *min {
                    continue;
                }
                if hash == *min {
                    continue;
                }
                match seen.entry(hash) {
                    Entry::Vacant(e) => {
                        e.insert(1usize);
                    }
                    Entry::Occupied(mut e) => {
                        let cnt = e.get_mut();
                        *cnt += 1;
                        if *cnt == self.params.count {
                            e.remove();
                            *min = hash;
                        }
                    }
                }
            }
            let num_empty = buckets.iter().filter(|x| **x == max_hash).count();
            if bound == max_hash || num_empty == 0 {
                let empty = if num_empty > 0 && self.params.filter_empty {
                    buckets
                        .chunks(64)
                        .map(|xs| {
                            xs.iter()
                                .enumerate()
                                .fold(0u64, |bits, (i, x)| bits | (((*x == max_hash) as u64) << i))
                        })
                        .collect()
                } else {
                    vec![]
                };
                let divisor = self.params.s as u64;
                let reduced = buckets
                    .iter()
                    .map(|x| {
                        if *x == max_hash {
                            max_hash
                        } else {
                            *x / divisor
                        }
                    })
                    .collect_vec();
                return BucketSketch {
                    hash_mode: self.params.hash_mode,
                    rc: self.params.rc,
                    k: self.params.k,
                    b: self.effective_b(),
                    seed: self.params.seed,
                    count: self.params.count,
                    empty,
                    buckets: BitSketch::new(self.effective_b(), &reduced),
                };
            }
            let new_factor = factor + factor.div_ceil(4);
            self.factor.fetch_max(new_factor, Relaxed);
        }
    }

    fn effective_b(&self) -> usize {
        match self.params.hash_mode {
            HashMode::Legacy32 => self.params.b.min(32),
            HashMode::NtHash64 => self.params.b,
        }
    }

    fn for_each_hash(&self, seqs: &[impl Sketchable], mut callback: impl FnMut(u64)) {
        match self.params.hash_mode {
            HashMode::Legacy32 => {
                let hasher = if self.params.rc {
                    EitherHasher::Rc(&self.rc_hasher)
                } else {
                    EitherHasher::Fwd(&self.fwd_hasher)
                };
                for &seq in seqs {
                    match hasher {
                        EitherHasher::Rc(hasher) => {
                            for hash in seq.legacy_hashes(hasher) {
                                if hash != u32::MAX {
                                    callback(hash as u64);
                                }
                            }
                        }
                        EitherHasher::Fwd(hasher) => {
                            for hash in seq.legacy_hashes(hasher) {
                                if hash != u32::MAX {
                                    callback(hash as u64);
                                }
                            }
                        }
                    }
                }
            }
            HashMode::NtHash64 => {
                for &seq in seqs {
                    let bases = seq.encoded_bases();
                    let mut it =
                        nthash64::NtHashIterator::new(bases, self.params.k, self.params.rc);
                    for hash in &mut it {
                        callback(hash);
                    }
                }
            }
        }
    }

    pub fn collect_up_to_bound(
        &self,
        seqs: &[impl Sketchable],
        mut bound: u64,
        out: &mut Vec<u64>,
        batch_size: usize,
        mut callback: impl FnMut(&mut Vec<u64>) -> u64,
    ) {
        out.clear();
        self.for_each_hash(seqs, |hash| {
            if hash <= bound {
                out.push(hash);
                if out.len() >= batch_size {
                    bound = callback(out);
                }
            }
        });
        callback(out);
    }
}

enum EitherHasher<'a> {
    Rc(&'a RcNtHasher),
    Fwd(&'a FwdNtHasher),
}

pub trait Sketchable: Copy {
    fn len(self) -> usize;
    fn legacy_hashes<H: KmerHasher>(self, hasher: &H) -> Vec<u32>;
    fn encoded_bases(self) -> Vec<u8>;
}

impl Sketchable for &[u8] {
    fn len(self) -> usize {
        Seq::len(&self)
    }

    fn legacy_hashes<H: KmerHasher>(self, hasher: &H) -> Vec<u32> {
        hasher.hash_kmers_scalar(self).collect()
    }

    fn encoded_bases(self) -> Vec<u8> {
        self.iter().map(|b| encode_base(*b)).collect()
    }
}

impl Sketchable for packed_seq::AsciiSeq<'_> {
    fn len(self) -> usize {
        Seq::len(&self)
    }

    fn legacy_hashes<H: KmerHasher>(self, hasher: &H) -> Vec<u32> {
        hasher.hash_kmers_scalar(self).collect()
    }

    fn encoded_bases(self) -> Vec<u8> {
        self.iter_bp().collect()
    }
}

impl Sketchable for packed_seq::PackedSeq<'_> {
    fn len(self) -> usize {
        Seq::len(&self)
    }

    fn legacy_hashes<H: KmerHasher>(self, hasher: &H) -> Vec<u32> {
        hasher.hash_kmers_scalar(self).collect()
    }

    fn encoded_bases(self) -> Vec<u8> {
        self.iter_bp().collect()
    }
}

impl<'s> Sketchable for PackedNSeq<'s> {
    fn len(self) -> usize {
        Seq::len(&self.seq)
    }

    fn legacy_hashes<H: KmerHasher>(self, hasher: &H) -> Vec<u32> {
        hasher.hash_valid_kmers_scalar(self).collect()
    }

    fn encoded_bases(self) -> Vec<u8> {
        self.seq
            .iter_bp()
            .zip(self.ambiguous.iter_bp())
            .map(|(base, amb)| if amb == 0 { base } else { 5 })
            .collect()
    }
}

fn encode_base(base: u8) -> u8 {
    let lower = base | 0x20;
    match lower {
        b'a' => 0,
        b'c' => 1,
        b't' | b'u' => 2,
        b'g' => 3,
        _ => 5,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use packed_seq::{PackedNSeqVec, PackedSeqVec, SeqVec};

    #[test]
    fn legacy_and_nt64_self_distance_zero() {
        let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
        for mode in [HashMode::Legacy32, HashMode::NtHash64] {
            let sketcher = SketchParams {
                alg: SketchAlg::Bucket,
                hash_mode: mode,
                rc: true,
                k: 7,
                s: 64,
                b: if mode == HashMode::NtHash64 { 64 } else { 32 },
                seed: 0,
                count: 0,
                coverage: 1,
                filter_empty: true,
                filter_out_n: false,
            }
            .build();
            let sketch = sketcher.sketch(seq.as_slice());
            assert_eq!(sketch.mash_distance(&sketch), 0.0);
        }
    }

    #[test]
    fn nthash64_skips_ambiguous_kmers() {
        let seq = PackedNSeqVec::from_ascii(b"ACGTNNNNACGTACGT");
        let sketcher = SketchParams {
            alg: SketchAlg::Bottom,
            hash_mode: HashMode::NtHash64,
            rc: true,
            k: 5,
            s: 8,
            b: 0,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: true,
        }
        .build();
        let sketch = match sketcher.sketch(seq.as_slice()) {
            Sketch::BottomSketch(sketch) => sketch,
            _ => unreachable!(),
        };
        assert!(sketch.bottom.iter().any(|x| *x != u64::MAX));
    }
}
