// benches/lattice_orthogonalization_4d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use rand::Rng;

fn gram_schmidt(vs: &[VecN<4>]) -> Vec<VecN<4>> {
    let mut us: Vec<VecN<4>> = vec![];
    for i in 0..vs.len() {
        let mut vi = vs[i].clone();
        for uj in &us {
            let proj = uj.clone().scale(vi.dot(uj) / uj.dot(uj));
            vi = vi - proj;
        }
        us.push(vi);
    }
    us
}

fn generate_lattice_basis(n: usize) -> Vec<VecN<4>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let base: [f64; 4] = std::array::from_fn(|_| rng.gen_range(-100.0..100.0));
            VecN::new(base)
        })
        .collect()
}

fn bench_lattice_orthogonalization_4d(c: &mut Criterion) {
    let input = generate_lattice_basis(4);

    c.bench_function("classical_gram_schmidt_4d", |b| {
        b.iter(|| {
            let out = gram_schmidt(black_box(&input));
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_lattice_orthogonalization_4d);
criterion_main!(benches);
