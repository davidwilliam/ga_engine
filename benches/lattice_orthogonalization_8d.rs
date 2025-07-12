// benches/lattice_orthogonalization_8d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use rand::Rng;

/// Classical Gram-Schmidt in 8D
fn gram_schmidt(vs: &[VecN<8>]) -> Vec<VecN<8>> {
  let mut us: Vec<VecN<8>> = vec![];  // <- type explicitly declared here
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

/// Generate 8D lattice basis with slight correlation
fn generate_lattice_basis(n: usize) -> Vec<VecN<8>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let base: [f64; 8] = std::array::from_fn(|_| rng.gen_range(-100.0..100.0));
            VecN::new(base)
        })
        .collect()
}

fn bench_lattice_orthogonalization(c: &mut Criterion) {
    let input = generate_lattice_basis(8);

    c.bench_function("classical_gram_schmidt_8d", |b| {
        b.iter(|| {
            let out = gram_schmidt(black_box(&input));
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_lattice_orthogonalization);
criterion_main!(benches);
