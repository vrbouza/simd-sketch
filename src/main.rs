use std::{
    fs::File,
    io::Seek,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use clap::Parser;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::info;
use packed_seq::{PackedNSeqVec, PackedSeqVec, SeqVec};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use simd_sketch::{BitSketch, HashMode, Sketch, SketchParams};

/// Compute the sketch distance between two fasta files.
#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// TODO: Support for writing sketches to disk.
#[derive(clap::Subcommand)]
enum Command {
    /// Takes paths to fasta files, and writes .ssketch files.
    Sketch {
        #[command(flatten)]
        params: SketchParams,
        /// Paths to (directories of) (gzipped) fasta files.
        paths: Vec<PathBuf>,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,
    },
    /// Compute the distance between two sequences.
    Dist {
        #[command(flatten)]
        params: SketchParams,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Takes paths to fasta files, and outputs a Phylip distance matrix to stdout.
    Triangle {
        #[command(flatten)]
        params: SketchParams,
        /// Paths to (directories of) (gzipped) fasta files or .ssketch files.
        /// If <path>.ssketch exists, it is automatically used.
        paths: Vec<PathBuf>,
        /// Write phylip distance matrix here, or default to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Save missing sketches to disk, as .ssketch files alongside the input.
        #[arg(long)]
        save_sketches: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Takes paths to fasta files, and writes .ssketch files.
    Classify {
        // Sketch args
        #[command(flatten)]
        params: SketchParams,
        /// Paths to directory of (gzipped) fasta files.
        #[arg(long)]
        targets: Vec<PathBuf>,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,

        /// Path to .fastq.gz metagenomic sample
        reads: PathBuf,
    },
}

const BINCODE_CONFIG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();
const EXTENSION: &str = "ssketch";
const SKETCH_VERSION_V1: usize = 1;
const SKETCH_VERSION_V2: usize = 2;

#[derive(bincode::Encode, bincode::Decode)]
pub struct VersionedSketchV2 {
    version: usize,
    sketch: Sketch,
}

#[derive(bincode::Encode, bincode::Decode)]
enum LegacySketch {
    BottomSketch(LegacyBottomSketch),
    BucketSketch(LegacyBucketSketch),
}

#[derive(bincode::Encode, bincode::Decode)]
struct LegacyBottomSketch {
    rc: bool,
    k: usize,
    seed: u32,
    count: usize,
    bottom: Vec<u32>,
}

#[derive(bincode::Encode, bincode::Decode)]
struct LegacyBucketSketch {
    rc: bool,
    k: usize,
    b: usize,
    seed: u32,
    count: usize,
    buckets: LegacyBitSketch,
    empty: Vec<u64>,
}

#[derive(bincode::Encode, bincode::Decode)]
enum LegacyBitSketch {
    B32(Vec<u32>),
    B16(Vec<u16>),
    B8(Vec<u8>),
    B1(Vec<u64>),
}

#[derive(bincode::Encode, bincode::Decode)]
struct VersionedSketchV1 {
    version: usize,
    sketch: LegacySketch,
}

impl From<LegacyBitSketch> for BitSketch {
    fn from(value: LegacyBitSketch) -> Self {
        match value {
            LegacyBitSketch::B32(v) => BitSketch::B32(v),
            LegacyBitSketch::B16(v) => BitSketch::B16(v),
            LegacyBitSketch::B8(v) => BitSketch::B8(v),
            LegacyBitSketch::B1(v) => BitSketch::B1(v),
        }
    }
}

impl From<LegacySketch> for Sketch {
    fn from(value: LegacySketch) -> Self {
        match value {
            LegacySketch::BottomSketch(sketch) => Sketch::BottomSketch(simd_sketch::BottomSketch {
                hash_mode: HashMode::Legacy32,
                rc: sketch.rc,
                k: sketch.k,
                seed: sketch.seed,
                count: sketch.count,
                bottom: sketch.bottom.into_iter().map(|x| x as u64).collect(),
            }),
            LegacySketch::BucketSketch(sketch) => Sketch::BucketSketch(simd_sketch::BucketSketch {
                hash_mode: HashMode::Legacy32,
                rc: sketch.rc,
                k: sketch.k,
                b: sketch.b,
                seed: sketch.seed,
                count: sketch.count,
                buckets: sketch.buckets.into(),
                empty: sketch.empty,
            }),
        }
    }
}

impl TryFrom<&Sketch> for LegacySketch {
    type Error = ();

    fn try_from(value: &Sketch) -> Result<Self, Self::Error> {
        match value {
            Sketch::BottomSketch(sketch) if sketch.hash_mode == HashMode::Legacy32 => {
                Ok(LegacySketch::BottomSketch(LegacyBottomSketch {
                    rc: sketch.rc,
                    k: sketch.k,
                    seed: sketch.seed,
                    count: sketch.count,
                    bottom: sketch.bottom.iter().map(|x| *x as u32).collect(),
                }))
            }
            Sketch::BucketSketch(sketch) if sketch.hash_mode == HashMode::Legacy32 => {
                let buckets = match &sketch.buckets {
                    BitSketch::B32(v) => LegacyBitSketch::B32(v.clone()),
                    BitSketch::B16(v) => LegacyBitSketch::B16(v.clone()),
                    BitSketch::B8(v) => LegacyBitSketch::B8(v.clone()),
                    BitSketch::B1(v) => LegacyBitSketch::B1(v.clone()),
                    BitSketch::B64(_) => return Err(()),
                };
                Ok(LegacySketch::BucketSketch(LegacyBucketSketch {
                    rc: sketch.rc,
                    k: sketch.k,
                    b: sketch.b,
                    seed: sketch.seed,
                    count: sketch.count,
                    buckets,
                    empty: sketch.empty.clone(),
                }))
            }
            _ => Err(()),
        }
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    // Initialize thread pool.
    let (Command::Sketch { threads, .. }
    | Command::Dist { threads, .. }
    | Command::Triangle { threads, .. }
    | Command::Classify { threads, .. }) = &args.command;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(*threads)
            .build_global()
            .unwrap();
    }

    let (params, paths) = match &args.command {
        Command::Dist {
            params,
            path_a,
            path_b,
            ..
        } => (params, vec![path_a.clone(), path_b.clone()]),
        Command::Sketch { params, paths, .. } | Command::Triangle { params, paths, .. } => {
            (params, collect_paths(&paths))
        }
        Command::Classify {
            params, targets, ..
        } => (params, collect_paths(&targets)),
    };

    let save_sketches = match &args.command {
        Command::Sketch { no_save, .. } => !no_save,
        Command::Classify { no_save, .. } => !no_save,
        Command::Dist { .. } => false,
        Command::Triangle { save_sketches, .. } => *save_sketches,
    };

    let q = paths.len();

    let sketcher = params.build();
    let params = sketcher.params();

    let style = indicatif::ProgressStyle::with_template(
        "{msg:.bold} [{elapsed_precise:.cyan}] {bar} {pos}/{len} ({percent:>3}%)",
    )
    .unwrap()
    .progress_chars("##-");

    let start = std::time::Instant::now();

    let num_sketched = AtomicUsize::new(0);
    let num_read = AtomicUsize::new(0);
    let num_written = AtomicUsize::new(0);
    let total_bytes = AtomicUsize::new(0);

    let sketches: Vec<_> = paths
        .par_iter()
        .progress_with_style(style.clone())
        .with_message("Sketching")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|path| {
            let read_sketch = |path| {
                num_read.fetch_add(1, Relaxed);
                let mut file = File::open(path).unwrap();
                let version: usize =
                    bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                file.seek(std::io::SeekFrom::Start(0)).unwrap();
                let sketch = match version {
                    SKETCH_VERSION_V1 => {
                        let VersionedSketchV1 { version, sketch } =
                            bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                        assert_eq!(version, SKETCH_VERSION_V1);
                        Sketch::from(sketch)
                    }
                    SKETCH_VERSION_V2 => {
                        let VersionedSketchV2 { version, sketch } =
                            bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                        assert_eq!(version, SKETCH_VERSION_V2);
                        sketch
                    }
                    _ => panic!("Unsupported sketch version: {version}."),
                };

                let mut sketch_params = sketch.to_params();
                sketch_params.filter_empty = params.filter_empty;
                if *params != sketch_params {
                    panic!(
                        "Sketch parameters do not match:\nCommand line: {params:?}\nOn disk:      {sketch_params:?}",
                    );
                }

                return sketch;
            };

            // Input path is a .ssketch file.
            if path.extension().is_some_and(|ext| ext == EXTENSION) {
                return read_sketch(path);
            }

            // Input path is a .fa, and the .fa.ssketch file exists.
            let ssketch_path = path.with_extension(EXTENSION);
            if ssketch_path.exists() {
                return read_sketch(&ssketch_path);
            }

            let mut reader = needletail::parse_fastx_file(&path).unwrap();

            let mut sketch;
            if params.filter_out_n {
                let mut ranges = vec![];
                let mut seq = PackedNSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let range = seq.push_ascii(&r.unwrap().seq());
                    size += range.len();
                    ranges.push(range);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs(&slices);
            } else {
                let mut ranges = vec![];
                let mut seq = PackedSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let range = seq.push_ascii(&r.unwrap().seq());
                    size += range.len();
                    ranges.push(range);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs(&slices);
            }
            num_sketched.fetch_add(1, Relaxed);

            if save_sketches {
                num_written.fetch_add(1, Relaxed);
                let mut writer = File::create(ssketch_path).unwrap();
                if params.hash_mode == HashMode::Legacy32 {
                    let versioned_sketch = VersionedSketchV1 {
                        version: SKETCH_VERSION_V1,
                        sketch: LegacySketch::try_from(&sketch).unwrap(),
                    };
                    bincode::encode_into_std_write(&versioned_sketch, &mut writer, BINCODE_CONFIG)
                        .unwrap();
                } else {
                    let versioned_sketch = VersionedSketchV2 {
                        version: SKETCH_VERSION_V2,
                        sketch,
                    };
                    bincode::encode_into_std_write(&versioned_sketch, &mut writer, BINCODE_CONFIG)
                        .unwrap();
                    sketch = versioned_sketch.sketch;
                }
            }

            sketch
        })
        .collect();
    let t_sketch = start.elapsed();

    info!(
        "Sketching {q} seqs took {t_sketch:?} ({:?} avg, {} MiB/s)",
        t_sketch / q as u32,
        total_bytes.into_inner() as f32 / t_sketch.as_secs_f32() / (1 << 20) as f32
    );
    let num_read = num_read.into_inner();
    let num_sketched = num_sketched.into_inner();
    let num_written = num_written.into_inner();
    if num_read > 0 {
        info!("Read {num_read} sketches from disk.");
    }
    if num_sketched > 0 {
        info!("Newly sketched {num_sketched} files.");
    }
    if num_written > 0 {
        info!("Wrote {num_written} sketches to disk.");
    }

    if matches!(args.command, Command::Sketch { .. }) {
        // If we are sketching, we are done.
        return;
    }
    if let Command::Classify { reads, .. } = &args.command {
        simd_sketch::classify::classify(&sketches, reads);
        return;
    }

    let num_pairs = q * (q - 1) / 2;
    let mut pairs = Vec::with_capacity(num_pairs);
    for i in 0..q {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    let start = std::time::Instant::now();
    let dists: Vec<_> = pairs
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Distances")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|(i, j)| sketches[i].mash_distance(&sketches[j]))
        .collect();
    let t_dist = start.elapsed();

    let cnt = q * (q - 1) / 2;
    info!(
        "Computing {cnt} dists took {t_dist:?} ({:?} avg)",
        t_dist / cnt.max(1) as u32
    );

    match &args.command {
        Command::Sketch { .. } => {
            unreachable!();
        }
        Command::Classify { .. } => {
            unreachable!();
        }
        Command::Dist { .. } => {
            println!("Distance: {:.4}", dists[0]);
            return;
        }
        Command::Triangle { output, .. } => {
            use std::io::Write;

            // Output Phylip triangle format.
            let mut out = Vec::new();
            writeln!(out, "{q}").unwrap();
            let mut d = dists.iter();
            for i in 0..q {
                write!(out, "{}", paths[i].to_str().unwrap()).unwrap();
                for _ in 0..i {
                    write!(out, "\t{:.7}", d.next().unwrap()).unwrap();
                }
                writeln!(out).unwrap();
            }

            match output {
                Some(output) => std::fs::write(output, out).unwrap(),
                None => println!("{}", str::from_utf8(&out).unwrap()),
            }
        }
    }
}

fn collect_paths(paths: &Vec<PathBuf>) -> Vec<PathBuf> {
    let mut res = vec![];
    for path in paths {
        if path.is_dir() {
            res.extend(path.read_dir().unwrap().map(|entry| entry.unwrap().path()));
        } else {
            res.push(path.clone());
        }
    }
    res.sort();

    let extensions = [
        "fa", "fasta", "fq", "fastq", "gz", "fasta.gz", "fq.gz", "fastq.gz",
    ];
    res.retain(|p| extensions.iter().any(|e| p.extension().unwrap() == *e));
    res
}
