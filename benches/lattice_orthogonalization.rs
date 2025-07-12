// benches/lattice_orthogonalization.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::vector::Vec3;

/// Classical Gram-Schmidt orthogonalization for 3 vectors.
fn classical_gram_schmidt(vs: &[Vec3; 3]) -> [Vec3; 3] {
    let mut u0 = vs[0];
    let mut u1 = vs[1] - u0.project_onto(&u0);
    let mut u2 = vs[2] - u0.project_onto(&u0) - u1.project_onto(&u1);
    [u0, u1, u2]
}

/// GA-style rejection-based orthogonalization for 3 vectors.
fn ga_orthogonalization(vs: &[Vec3; 3]) -> [Vec3; 3] {
    let u0 = vs[0];
    let u1 = vs[1].reject_from(&u0);
    let u2 = vs[2].reject_from(&u0).reject_from(&u1);
    [u0, u1, u2]
}

fn bench_orthogonalization(c: &mut Criterion) {
    // Hard lattice basis — almost linearly dependent
    let basis = [
        Vec3::new(100.0, 1.0, 0.5),
        Vec3::new(99.0, 1.5, 0.3),
        Vec3::new(98.0, 1.8, 0.4),
    ];

    c.bench_function("classical_gram_schmidt", |b| {
        b.iter(|| {
            let ortho = classical_gram_schmidt(black_box(&basis));
            black_box(ortho)
        })
    });

    c.bench_function("ga_orthogonalization", |b| {
        b.iter(|| {
            let ortho = ga_orthogonalization(black_box(&basis));
            black_box(ortho)
        })
    });
}

criterion_group!(benches, bench_orthogonalization);
criterion_main!(benches);
