// benches/svp_8d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use ga_engine::nd::multivector::Multivector;
use rand::Rng;

/// Generate a random lattice basis in 8D with `n` vectors
fn generate_basis(n: usize) -> Vec<VecN<8>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let arr = std::array::from_fn(|_| rng.gen_range(-50.0..50.0));
            VecN::new(arr)
        })
        .collect()
}

/// Classical brute-force SVP: find pair with shortest Euclidean distance
fn classical_svp(vectors: &[VecN<8>]) -> (VecN<8>, VecN<8>) {
    let mut min_dist = f64::MAX;
    let mut best = (vectors[0].clone(), vectors[1].clone());
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let diff = vectors[i].clone() - vectors[j].clone();
            let norm = diff.norm();
            if norm < min_dist {
                min_dist = norm;
                best = (vectors[i].clone(), vectors[j].clone());
            }
        }
    }
    best
}

/// GA-based SVP using multivectors
fn ga_svp(vectors: &[VecN<8>]) -> (Multivector<8>, Multivector<8>) {
    let mvectors: Vec<_> = vectors
        .iter()
        .map(|v| {
            let mut data = vec![0.0; 1 << 8];
            for i in 0..8 {
                data[1 << i] = v.data[i];
            }
            Multivector::<8>::new(data)
        })
        .collect();

    let mut min_norm = f64::MAX;
    let mut best = (mvectors[0].clone(), mvectors[1].clone());

    for i in 0..mvectors.len() {
        for j in (i + 1)..mvectors.len() {
            let diff = mvectors[i].clone() - mvectors[j].clone();
            let norm = diff.data.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < min_norm {
                min_norm = norm;
                best = (mvectors[i].clone(), mvectors[j].clone());
            }
        }
    }
    best
}

fn bench_svp_8d(c: &mut Criterion) {
    let vectors = generate_basis(50);

    c.bench_function("classical_svp_8d", |b| {
        b.iter(|| {
            let _ = classical_svp(black_box(&vectors));
        });
    });

    c.bench_function("ga_svp_8d", |b| {
        b.iter(|| {
            let _ = ga_svp(black_box(&vectors));
        });
    });
}

criterion_group!(benches, bench_svp_8d);
criterion_main!(benches);
