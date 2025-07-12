// benches/ga_orthogonalization_16d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::multivector::Multivector;
use ga_engine::nd::types::Scalar;
use rand::Rng;

const D: usize = 16;
type MV = Multivector<D>;

/// Generate a 16D random multivector with only vector components (blades 1..D)
fn random_vector_mv() -> MV {
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0; 1 << D];
    for i in 0..D {
        data[1 << i] = rng.gen_range(-100.0..100.0);
    }
    MV::new(data)
}

/// Project mv onto axis using GA: proj = axis * (mv·axis) / (axis·axis)
fn ga_project(v: &MV, axis: &MV) -> MV {
    let num = v.clone().gp(axis);
    let denom = axis.clone().gp(axis);
    axis.clone() * (num.data[0] / denom.data[0])
}

/// GA-based Gram-Schmidt orthogonalization in 16D
fn ga_orthogonalize(vs: &[MV]) -> Vec<MV> {
    let mut us = Vec::new();
    for i in 0..vs.len() {
        let mut vi = vs[i].clone();
        for uj in &us {
            let proj = ga_project(&vi, uj);
            vi = vi - proj;
        }
        us.push(vi);
    }
    us
}

fn bench_ga_orthogonalization_16d(c: &mut Criterion) {
    let input: Vec<MV> = (0..D).map(|_| random_vector_mv()).collect();

    c.bench_function("ga_orthogonalize_16d", |b| {
        b.iter(|| {
            let out = ga_orthogonalize(black_box(&input));
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_ga_orthogonalization_16d);
criterion_main!(benches);
