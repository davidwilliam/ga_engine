//! CRYSTALS-Kyber Batch Encryption Benchmarks: Classical vs GA
//!
//! This benchmark demonstrates that GA-based approaches can accelerate
//! batched Kyber operations, the NIST-selected post-quantum encryption standard.
//!
//! ## Background
//!
//! CRYSTALS-Kyber is the NIST-selected standard for post-quantum encryption (2022).
//! The core operation is matrix-vector multiplication A·s in polynomial rings.
//!
//! **Challenge**: Kyber-512 uses small 2×2 matrices (too small for 8×8 GA speedup)
//!
//! **Solution**: Batch 4 encryptions into a single 8×8 block-diagonal operation
//!
//! ## Approaches Compared
//!
//! 1. **Classical Single**: Encrypt one message at a time (baseline)
//! 2. **Classical Batch-4**: Encrypt 4 messages separately (naive batching)
//! 3. **GA Batch-4**: Encrypt 4 messages using 8×8 GA acceleration
//!
//! ## Expected Results
//!
//! Based on NTRU success (2.57× speedup on N=8):
//! - **Target**: 2.0-2.5× speedup on batch-4 vs classical batch-4
//! - **vs Recent work**: "Efficient Batch Algorithms" (2024) got 12-30% improvement
//! - **Our goal**: 2× improvement (100% better than their 12-30%)
//!
//! ## References
//!
//! - NIST PQC Standardization (2022)
//! - "Efficient Batch Algorithms for Post-Quantum Crystals" (2024)
//! - Our NTRU results: 2.57× speedup on 8×8 operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ga_engine::kyber::params::KyberParams;
use ga_engine::kyber::classical::{kyber_keygen, create_test_message};
use ga_engine::kyber::batch::{
    kyber_encrypt_batch_classical_4,
    create_test_messages_batch,
};
use ga_engine::kyber::ga_batch::kyber_encrypt_batch_ga_4;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Benchmark single encryption (baseline)
fn bench_single_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("kyber_single_encryption");
    group.throughput(Throughput::Elements(1));

    let params = KyberParams::KYBER512;
    let mut rng = StdRng::seed_from_u64(42);

    let (pk, _sk) = kyber_keygen(params, &mut rng);
    let message = create_test_message(params, &mut rng);

    group.bench_function("kyber512_encrypt_single", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let ct = ga_engine::kyber::classical::kyber_encrypt_single(
                black_box(&pk),
                black_box(&message),
                &mut local_rng,
            );
            black_box(ct)
        })
    });

    group.finish();
}

/// Benchmark batch-4 encryption: Classical vs GA
///
/// **This is the key comparison!**
fn bench_batch_4_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kyber_batch_4_comparison");
    group.throughput(Throughput::Elements(4));

    let params = KyberParams::KYBER512;
    let mut rng = StdRng::seed_from_u64(42);

    let (pk, _sk) = kyber_keygen(params, &mut rng);

    let messages = [
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
    ];

    // Classical batch-4 (baseline)
    group.bench_function("classical_batch_4", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let cts = kyber_encrypt_batch_classical_4(
                black_box(&pk),
                black_box(&messages),
                &mut local_rng,
            );
            black_box(cts)
        })
    });

    // GA-accelerated batch-4 (our optimization)
    group.bench_function("ga_batch_4", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let cts = kyber_encrypt_batch_ga_4(
                black_box(&pk),
                black_box(&messages),
                &mut local_rng,
            );
            black_box(cts)
        })
    });

    group.finish();
}

/// Benchmark different batch sizes
fn bench_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kyber_batch_scaling");

    let params = KyberParams::KYBER512;
    let mut rng = StdRng::seed_from_u64(42);

    let (pk, _sk) = kyber_keygen(params, &mut rng);

    for &batch_size in &[1, 4, 8, 16] {
        group.throughput(Throughput::Elements(batch_size as u64));

        let messages = create_test_messages_batch(params, batch_size, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("classical", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let mut local_rng = StdRng::seed_from_u64(42);
                    let cts = ga_engine::kyber::batch::kyber_encrypt_batch_classical(
                        black_box(&pk),
                        black_box(&messages),
                        &mut local_rng,
                    );
                    black_box(cts)
                })
            },
        );

        // Only bench GA for batch-4 (our optimized size)
        if batch_size == 4 {
            let messages_arr = [
                messages[0].clone(),
                messages[1].clone(),
                messages[2].clone(),
                messages[3].clone(),
            ];

            group.bench_with_input(
                BenchmarkId::new("ga_optimized", batch_size),
                &batch_size,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut local_rng = StdRng::seed_from_u64(42);
                        let cts = kyber_encrypt_batch_ga_4(
                            black_box(&pk),
                            black_box(&messages_arr),
                            &mut local_rng,
                        );
                        black_box(cts)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Comprehensive benchmark: All methods, detailed statistics
fn bench_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("kyber_comprehensive");
    group.sample_size(100);

    let params = KyberParams::KYBER512;
    let mut rng = StdRng::seed_from_u64(42);

    let (pk, _sk) = kyber_keygen(params, &mut rng);

    let messages = [
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
        create_test_message(params, &mut rng),
    ];

    // Single encryption
    group.bench_function("01_single_encryption", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let ct = ga_engine::kyber::classical::kyber_encrypt_single(
                black_box(&pk),
                black_box(&messages[0]),
                &mut local_rng,
            );
            black_box(ct)
        })
    });

    // Classical batch-4
    group.bench_function("02_classical_batch_4", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let cts = kyber_encrypt_batch_classical_4(
                black_box(&pk),
                black_box(&messages),
                &mut local_rng,
            );
            black_box(cts)
        })
    });

    // GA batch-4
    group.bench_function("03_ga_batch_4", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let cts = kyber_encrypt_batch_ga_4(
                black_box(&pk),
                black_box(&messages),
                &mut local_rng,
            );
            black_box(cts)
        })
    });

    group.finish();
}

/// Real-world scenario: TLS server encrypting multiple session keys
fn bench_tls_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("kyber_tls_scenario");
    group.throughput(Throughput::Elements(100));

    let params = KyberParams::KYBER512;
    let mut rng = StdRng::seed_from_u64(42);

    let (pk, _sk) = kyber_keygen(params, &mut rng);

    // Simulate 100 session key encryptions (typical TLS server load)
    let messages = create_test_messages_batch(params, 100, &mut rng);

    group.bench_function("classical_100_encryptions", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let cts = ga_engine::kyber::batch::kyber_encrypt_batch_classical(
                black_box(&pk),
                black_box(&messages),
                &mut local_rng,
            );
            black_box(cts)
        })
    });

    // GA batch approach: process in chunks of 4
    group.bench_function("ga_100_encryptions_batched", |bencher| {
        bencher.iter(|| {
            let mut local_rng = StdRng::seed_from_u64(42);
            let mut all_cts = Vec::with_capacity(100);

            // Process in batches of 4
            for chunk_start in (0..100).step_by(4) {
                let batch = [
                    messages[chunk_start].clone(),
                    messages[chunk_start + 1].clone(),
                    messages[chunk_start + 2].clone(),
                    messages[chunk_start + 3].clone(),
                ];

                let cts = kyber_encrypt_batch_ga_4(
                    black_box(&pk),
                    black_box(&batch),
                    &mut local_rng,
                );

                all_cts.extend_from_slice(&cts);
            }

            black_box(all_cts)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_encryption,
    bench_batch_4_comparison,
    bench_batch_scaling,
    bench_comprehensive,
    bench_tls_scenario,
);

criterion_main!(benches);
