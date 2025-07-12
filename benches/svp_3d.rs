// benches/svp_3d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::vector::Vec3;
use ga_engine::bivector::Bivector3;
use ga_engine::multivector::Multivector3;
use ga_engine::prelude::*;
use rand::Rng;

/// Generate a random 3D lattice basis
fn generate_lattice_basis() -> [Vec3; 3] {
    let mut rng = rand::thread_rng();
    [0, 1, 2].map(|_| Vec3::new(rng.gen_range(-50.0..50.0), rng.gen_range(-50.0..50.0), rng.gen_range(-50.0..50.0)))
}

/// Classical method: return shortest vector using naive norm check
fn classical_svp(basis: &[Vec3; 3]) -> Vec3 {
    let mut shortest = basis[0];
    let mut shortest_len = basis[0].norm();
    for v in basis {
        let len = v.norm();
        if len < shortest_len {
            shortest = *v;
            shortest_len = len;
        }
    }
    shortest
}

/// GA method: use bivectors and wedge products to find minimal vector
fn ga_svp(basis: &[Vec3; 3]) -> Vec3 {
    let mut shortest = basis[0];
    let mut min_volume = Bivector3::from_wedge(basis[0], basis[1]).norm();

    for i in 0..3 {
        for j in (i+1)..3 {
            let area = Bivector3::from_wedge(basis[i], basis[j]).norm();
            if area < min_volume {
                shortest = if basis[i].norm() < basis[j].norm() { basis[i] } else { basis[j] };
                min_volume = area;
            }
        }
    }
    shortest
}

fn bench_svp(c: &mut Criterion) {
    let basis = generate_lattice_basis();

    c.bench_function("classical_svp_3d", |b| {
        b.iter(|| {
            let res = classical_svp(black_box(&basis));
            black_box(res);
        })
    });

    c.bench_function("ga_svp_3d", |b| {
        b.iter(|| {
            let res = ga_svp(black_box(&basis));
            black_box(res);
        })
    });
}

criterion_group!(benches, bench_svp);
criterion_main!(benches);
