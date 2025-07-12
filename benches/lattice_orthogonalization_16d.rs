// benches/lattice_orthogonalization_16d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use rand::Rng;

/// Classical Gram-Schmidt in 16D
fn gram_schmidt(vs: &[VecN<16>]) -> Vec<VecN<16>> {
    let mut us: Vec<VecN<16>> = vec![];
    for i in 0..vs.len() {
        let mut vi = vs[i].clone();
        for uj in &us {
            let dot_uj = uj.dot(uj);
            if dot_uj != 0.0 {
                let proj = uj.clone().scale(vi.dot(uj) / dot_uj);
                vi = vi - proj;
            }
        }
        us.push(vi);
    }
    us
}

/// Generate 16D lattice basis with slight correlation
fn generate_lattice_basis(n: usize) -> Vec<VecN<16>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let base: [f64; 16] = std::array::from_fn(|_| rng.gen_range(-100.0..100.0));
            VecN::new(base)
        })
        .collect()
}

fn bench_lattice_orthogonalization(c: &mut Criterion) {
    let input = generate_lattice_basis(16);

    c.bench_function("classical_gram_schmidt_16d", |b| {
        b.iter(|| {
            let out = gram_schmidt(black_box(&input));
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_lattice_orthogonalization);
criterion_main!(benches);
