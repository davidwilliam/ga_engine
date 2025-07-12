use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use rand::prelude::*;

const N: usize = 8;
const VEC_COUNT: usize = 8;

type V = VecN<N>;

fn random_vecs(seed: u64) -> Vec<V> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..VEC_COUNT)
        .map(|_| {
            let data = core::array::from_fn(|_| rng.gen_range(-10.0..10.0));
            V::new(data)
        })
        .collect()
}

fn classical_gram_schmidt(vs: &[V]) -> Vec<V> {
    let mut us: Vec<V> = Vec::with_capacity(vs.len());
    for v in vs {
        let mut proj = V::new([0.0; N]);
        for u in &us {
            let coeff = v.dot(u) / u.dot(u);
            proj = proj + u.clone() * coeff;
        }
        us.push(v.clone() - proj);
    }
    us
}

fn bench_classical(c: &mut Criterion) {
    let vs = random_vecs(42);
    c.bench_function("classical_gram_schmidt_8d", |b| {
        b.iter(|| classical_gram_schmidt(black_box(&vs)))
    });
}

criterion_group!(benches, bench_classical);
criterion_main!(benches);
