// benches/svp_32d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use rand::Rng;

/// Generate a random 32D lattice basis with bounded entries
fn generate_lattice_basis(n: usize) -> Vec<VecN<32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let base: [f64; 32] = std::array::from_fn(|_| rng.gen_range(-50.0..50.0));
            VecN::new(base)
        })
        .collect()
}

/// Classical Gram-Schmidt rejection step
fn classical_rejection_step(vs: &[VecN<32>]) -> VecN<32> {
    let mut v = vs[0].clone();
    for u in &vs[1..] {
        let proj = u.clone().scale(v.dot(u) / u.dot(u));
        v = v - proj;
    }
    v
}

fn bench_svp_32d(c: &mut Criterion) {
    let mut input = generate_lattice_basis(4); // 4-vector basis

    c.bench_function("svp_classical_rejection_32d", |b| {
        b.iter(|| {
            let r = classical_rejection_step(black_box(&input));
            black_box(r);
        })
    });
}

criterion_group!(benches, bench_svp_32d);
criterion_main!(benches);
