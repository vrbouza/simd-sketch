use packed_seq::{BitSeqVec, PackedNSeq, PackedNSeqVec, PackedSeqVec, SeqVec};
use simd_sketch::{BitSketch, HashMode, Sketch, SketchAlg, SketchParams};

fn params(hash_mode: HashMode, alg: SketchAlg, b: usize) -> SketchParams {
    SketchParams {
        alg,
        hash_mode,
        rc: true,
        k: 7,
        s: 64,
        b,
        seed: 0,
        count: 0,
        coverage: 1,
        filter_empty: true,
        filter_out_n: false,
    }
}

#[test]
fn bucket_sketch_uses_full_width_storage_in_nthash64_mode() {
    let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
    let sketch = params(HashMode::NtHash64, SketchAlg::Bucket, 64)
        .build()
        .sketch(seq.as_slice());
    let Sketch::BucketSketch(sketch) = sketch else {
        panic!("expected bucket sketch")
    };
    assert!(matches!(sketch.buckets, BitSketch::B64(_)));
}

#[test]
fn nthash64_bucket_sketch_still_supports_b16_storage() {
    let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
    let sketch = params(HashMode::NtHash64, SketchAlg::Bucket, 16)
        .build()
        .sketch(seq.as_slice());
    let Sketch::BucketSketch(sketch) = sketch else {
        panic!("expected bucket sketch")
    };
    assert!(matches!(sketch.buckets, BitSketch::B16(_)));
}

#[test]
fn legacy32_bottom_sketch_rounds_to_u64_storage() {
    let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
    let sketch = params(HashMode::Legacy32, SketchAlg::Bottom, 0)
        .build()
        .sketch(seq.as_slice());
    let Sketch::BottomSketch(sketch) = sketch else {
        panic!("expected bottom sketch")
    };
    assert_eq!(sketch.hash_mode, HashMode::Legacy32);
    assert_eq!(sketch.bottom.len(), 64);
    assert!(sketch.bottom.iter().all(|x| *x <= u32::MAX as u64));
}

#[test]
fn nthash64_filtering_keeps_self_distance_zero_with_ambiguous_input() {
    let seq = PackedNSeqVec::from_ascii(b"ACGTNNNNACGTACGTNNNNACGTACGT");
    let sketcher = SketchParams {
        filter_out_n: true,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 64)
    }
    .build();
    let sketch = sketcher.sketch(seq.as_slice());
    assert_eq!(sketch.mash_distance(&sketch), 0.0);
}

#[test]
fn nthash64_bottom_sketch_with_ambiguous_input_is_deterministic() {
    let seq = PackedNSeqVec::from_ascii(b"NNNNACGTACGTNNNNACGTNACGTACGTNNNN");
    let sketcher = SketchParams {
        filter_out_n: true,
        ..params(HashMode::NtHash64, SketchAlg::Bottom, 0)
    }
    .build();

    let sketch_a = sketcher.sketch(seq.as_slice());
    let sketch_b = sketcher.sketch(seq.as_slice());

    let Sketch::BottomSketch(sketch_a) = sketch_a else {
        panic!("expected bottom sketch")
    };
    let Sketch::BottomSketch(sketch_b) = sketch_b else {
        panic!("expected bottom sketch")
    };
    assert_eq!(sketch_a.bottom, sketch_b.bottom);
}

#[test]
fn nthash64_bucket_sketch_with_ambiguous_input_is_deterministic() {
    let seq = PackedNSeqVec::from_ascii(b"NNNNACGTACGTNNNNACGTNACGTACGTNNNN");
    let sketcher = SketchParams {
        filter_out_n: true,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 64)
    }
    .build();

    let sketch_a = sketcher.sketch(seq.as_slice());
    let sketch_b = sketcher.sketch(seq.as_slice());

    let Sketch::BucketSketch(sketch_a) = sketch_a else {
        panic!("expected bucket sketch")
    };
    let Sketch::BucketSketch(sketch_b) = sketch_b else {
        panic!("expected bucket sketch")
    };

    match (&sketch_a.buckets, &sketch_b.buckets) {
        (BitSketch::B64(a), BitSketch::B64(b)) => assert_eq!(a, b),
        _ => panic!("expected B64 bucket sketch"),
    }
    assert_eq!(sketch_a.empty, sketch_b.empty);
}

#[test]
fn nthash64_multi_record_sketching_is_deterministic() {
    let seq_a = PackedNSeqVec::from_ascii(b"ACGTNNNNACGTACGTNNNNACGTACGT");
    let seq_b = PackedNSeqVec::from_ascii(b"NNNNACGTACGTNNNNACGTNACGTACGTNNNN");
    let seqs = [seq_a.as_slice(), seq_b.as_slice()];
    let sketcher = SketchParams {
        filter_out_n: true,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 16)
    }
    .build();

    let sketch_a = sketcher.sketch_seqs(&seqs);
    let sketch_b = sketcher.sketch_seqs(&seqs);

    let Sketch::BucketSketch(sketch_a) = sketch_a else {
        panic!("expected bucket sketch")
    };
    let Sketch::BucketSketch(sketch_b) = sketch_b else {
        panic!("expected bucket sketch")
    };
    match (&sketch_a.buckets, &sketch_b.buckets) {
        (BitSketch::B16(a), BitSketch::B16(b)) => assert_eq!(a, b),
        _ => panic!("expected B16 bucket sketch"),
    }
    assert_eq!(sketch_a.empty, sketch_b.empty);
}

#[test]
fn nthash64_bucket_count_filtering_ignores_coverage() {
    let seq = PackedSeqVec::from_ascii(b"ACGTTGCATGTCAGTACGATCGTACG");
    let seqs = [seq.as_slice(), seq.as_slice(), seq.as_slice()];
    let params = SketchParams {
        count: 3,
        coverage: 1,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 16)
    };
    let sketch_a = params.build().sketch_seqs(&seqs);
    let sketch_b = SketchParams {
        coverage: 100,
        ..params
    }
    .build()
    .sketch_seqs(&seqs);

    let Sketch::BucketSketch(sketch_a) = sketch_a else {
        panic!("expected bucket sketch")
    };
    let Sketch::BucketSketch(sketch_b) = sketch_b else {
        panic!("expected bucket sketch")
    };
    match (&sketch_a.buckets, &sketch_b.buckets) {
        (BitSketch::B16(a), BitSketch::B16(b)) => {
            assert_eq!(a, b);
            assert!(a.iter().any(|x| *x != u16::MAX));
        }
        _ => panic!("expected B16 bucket sketch"),
    }
    assert_eq!(sketch_a.empty, sketch_b.empty);
}

#[test]
fn nthash64_bucket_count_filtering_matches_repeated_records() {
    let seq = PackedSeqVec::from_ascii(b"ACGTTGCATGTCAGTACGATCGTACG");
    let repeated = [seq.as_slice(), seq.as_slice(), seq.as_slice()];
    let counted = SketchParams {
        count: 3,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 16)
    }
    .build()
    .sketch_seqs(&repeated);
    let uncounted = SketchParams {
        count: 1,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 16)
    }
    .build()
    .sketch(seq.as_slice());

    let Sketch::BucketSketch(counted) = counted else {
        panic!("expected bucket sketch")
    };
    let Sketch::BucketSketch(uncounted) = uncounted else {
        panic!("expected bucket sketch")
    };
    match (&counted.buckets, &uncounted.buckets) {
        (BitSketch::B16(a), BitSketch::B16(b)) => assert_eq!(a, b),
        _ => panic!("expected B16 bucket sketch"),
    }
    assert_eq!(counted.empty, uncounted.empty);
}

#[test]
fn nthash64_bucket_count_filtering_skips_ambiguous_windows() {
    let bases = PackedSeqVec::from_ascii(b"ACGTTGCA");
    let clean_flags = BitSeqVec::from_ascii(b"AAAAAAAA");
    let masked_flags = BitSeqVec::from_ascii(b"NNNNNNNN");
    let clean = PackedNSeq {
        seq: bases.as_slice(),
        ambiguous: clean_flags.as_slice(),
    };
    let masked = PackedNSeq {
        seq: bases.as_slice(),
        ambiguous: masked_flags.as_slice(),
    };
    let sketch = SketchParams {
        k: 8,
        s: 16,
        count: 2,
        filter_out_n: true,
        ..params(HashMode::NtHash64, SketchAlg::Bucket, 16)
    }
    .build()
    .sketch_seqs(&[clean, masked]);

    let Sketch::BucketSketch(sketch) = sketch else {
        panic!("expected bucket sketch")
    };
    match &sketch.buckets {
        BitSketch::B16(buckets) => assert!(buckets.iter().all(|x| *x == u16::MAX)),
        _ => panic!("expected B16 bucket sketch"),
    }
    assert!(sketch.empty.iter().any(|x| *x != 0));
}

#[test]
fn legacy32_bucket_count_filtering_is_deterministic() {
    let seq = PackedSeqVec::from_ascii(b"ACGTTGCATGTCAGTACGATCGTACG");
    let seqs = [seq.as_slice(), seq.as_slice()];
    let sketcher = SketchParams {
        count: 2,
        ..params(HashMode::Legacy32, SketchAlg::Bucket, 32)
    }
    .build();

    let sketch_a = sketcher.sketch_seqs(&seqs);
    let sketch_b = sketcher.sketch_seqs(&seqs);
    let Sketch::BucketSketch(sketch_a) = sketch_a else {
        panic!("expected bucket sketch")
    };
    let Sketch::BucketSketch(sketch_b) = sketch_b else {
        panic!("expected bucket sketch")
    };
    match (&sketch_a.buckets, &sketch_b.buckets) {
        (BitSketch::B32(a), BitSketch::B32(b)) => {
            assert_eq!(a, b);
            assert!(a.iter().any(|x| *x != u32::MAX));
        }
        _ => panic!("expected B32 bucket sketch"),
    }
    assert_eq!(sketch_a.empty, sketch_b.empty);
}
