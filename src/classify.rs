#![cfg(not(target_arch = "wasm32"))]

use std::{collections::HashMap, path::Path};

use itertools::Itertools;
use log::info;

use crate::{BitSketch, DnaInputOptions, Sketch, load_dna_file};

pub fn classify(sketches: &[Sketch], reads: &Path, input: &DnaInputOptions) {
    let mut params = sketches[0].to_params();
    params.filter_out_n = true;

    let mut max_sketch = vec![0u64; params.s];
    let mut counts = vec![HashMap::<u64, u32>::new(); params.s];
    let mut occ = HashMap::<u64, Vec<u16>>::new();
    for (j, sketch) in sketches.iter().enumerate() {
        let Sketch::BucketSketch(bucket_sketch) = sketch else {
            panic!("classify requires bucket sketches")
        };
        match &bucket_sketch.buckets {
            BitSketch::B64(buckets) => {
                assert_eq!(buckets.len(), params.s);
                for i in 0..params.s {
                    let v = buckets[i]
                        .saturating_mul(params.s as u64)
                        .saturating_add(i as u64);
                    max_sketch[i] = max_sketch[i].max(v);
                    *counts[i].entry(v).or_default() += 1;
                    occ.entry(v).or_default().push(j as u16);
                }
            }
            BitSketch::B32(buckets) => {
                assert_eq!(buckets.len(), params.s);
                for i in 0..params.s {
                    let v = buckets[i] as u64 * params.s as u64 + i as u64;
                    max_sketch[i] = max_sketch[i].max(v);
                    *counts[i].entry(v).or_default() += 1;
                    occ.entry(v).or_default().push(j as u16);
                }
            }
            _ => panic!("classify requires full-width bucket sketches"),
        }
    }

    let avg = max_sketch.iter().sum::<u64>() / max_sketch.len().max(1) as u64;
    let threshold = *max_sketch.iter().max().unwrap_or(&0);
    eprintln!("avg: {avg:?}");
    eprintln!("max: {threshold:?}");

    let lens = counts.iter().map(|s| s.len()).collect_vec();
    eprintln!("#distinct kmers in each bucket: {lens:?}");

    let mut bucket0_counts = counts[0].values().collect_vec();
    bucket0_counts.sort();
    eprintln!("kmer counts for bucket 0: {bucket0_counts:?}");

    let mut occ_flat: Vec<u16> = vec![];
    let mut ranges = HashMap::new();
    for (hash, vals) in &occ {
        let l0 = occ_flat.len() as u32;
        occ_flat.extend_from_slice(vals);
        let l1 = occ_flat.len() as u32;
        ranges.insert(*hash, l0..l1);
    }

    let mut counts = vec![0; sketches.len()];
    let mut per_bucket_counts = vec![HashMap::<u64, u32>::new(); sketches.len()];

    info!("Params: {params:?}");
    let sketcher = params.build();

    let seq = load_dna_file(reads, input);
    let mut read_hashes = vec![];
    sketcher.collect_up_to_bound(
        &[seq.as_slice()],
        threshold,
        &mut read_hashes,
        8000,
        |hashes| {
            for h in &*hashes {
                let Some(range) = ranges.get(h) else {
                    continue;
                };
                for &j in &occ_flat[range.start as usize..range.end as usize] {
                    counts[j as usize] += 1;
                    *per_bucket_counts[j as usize].entry(*h).or_default() += 1;
                }
            }
            hashes.clear();
            threshold
        },
    );

    counts.sort();
    info!("Number of matching kmers per target: {counts:?}");
    per_bucket_counts.sort_by_cached_key(|hm| hm.values().sum::<u32>());
    let per_bucket_counts = per_bucket_counts
        .iter()
        .map(|x| {
            let mut vals = x.values().copied().collect_vec();
            vals.sort();
            vals.iter()
                .chunk_by(|x| **x)
                .into_iter()
                .map(|(key, g)| (key, g.count()))
                .collect_vec()
        })
        .collect_vec();
    info!("Number of matching kmers per target and bucket: {per_bucket_counts:?}");
}
