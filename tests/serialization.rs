use packed_seq::{PackedSeqVec, SeqVec};
use simd_sketch::{HashMode, Sketch, SketchAlg, SketchParams};

const BINCODE_CONFIG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();

#[derive(bincode::Encode, bincode::Decode)]
struct VersionedSketchV2 {
    version: usize,
    sketch: Sketch,
}

#[test]
fn v2_sketch_roundtrip_preserves_hash_mode() {
    let seq = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
    let sketch = SketchParams {
        alg: SketchAlg::Bucket,
        hash_mode: HashMode::NtHash64,
        rc: true,
        k: 7,
        s: 64,
        b: 64,
        seed: 0,
        count: 0,
        coverage: 1,
        filter_empty: true,
        filter_out_n: false,
    }
    .build()
    .sketch(seq.as_slice());

    let encoded =
        bincode::encode_to_vec(VersionedSketchV2 { version: 2, sketch }, BINCODE_CONFIG).unwrap();
    let (decoded, _): (VersionedSketchV2, usize) =
        bincode::decode_from_slice(&encoded, BINCODE_CONFIG).unwrap();
    let Sketch::BucketSketch(decoded) = decoded.sketch else {
        panic!("expected bucket sketch")
    };
    assert_eq!(decoded.hash_mode, HashMode::NtHash64);
    assert_eq!(decoded.b, 64);
}
