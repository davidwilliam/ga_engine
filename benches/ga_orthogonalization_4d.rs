// benches/ga_orthogonalization_4d.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::multivector::Multivector;
use rand::Rng;

type MV = Multivector<4>;

/// Convert a 4D vector to a GA multivector (as pure vector blade)
fn vec_to_mv(v: [f64; 4]) -> MV {
    let mut data = vec![0.0; 16];
    data[1] = v[0]; // e1
    data[2] = v[1]; // e2
    data[4] = v[2]; // e3
    data[8] = v[3]; // e4
    MV::new(data)
}

/// Extract vector part from multivector
fn mv_to_vec(mv: &MV) -> [f64; 4] {
    [mv.data[1], mv.data[2], mv.data[4], mv.data[8]]
}

/// GA Gram-Schmidt in 4D
fn ga_orthogonalize(vs: &[MV]) -> Vec<MV> {
    let mut us: Vec<MV> = vec![];
    for v in vs {
        let mut u = v.clone();
        for uj in &us {
            let num = uj.clone().gp(&u);
            let denom = uj.clone().gp(uj);
            let scale = num.data[0] / denom.data[0];
            let proj = uj.clone() * scale;
            u = u.clone() - proj;
        }
        us.push(u);
    }
    us
}

/// Generate 4D lattice basis
fn generate_4d_lattice(n: usize) -> Vec<MV> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let arr: [f64; 4] = std::array::from_fn(|_| rng.gen_range(-100.0..100.0));
            vec_to_mv(arr)
        })
        .collect()
}

fn bench_ga_orthogonalization_4d(c: &mut Criterion) {
    let input = generate_4d_lattice(4);

    c.bench_function("ga_orthogonalize_4d", |b| {
        b.iter(|| {
            let out = ga_orthogonalize(black_box(&input));
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_ga_orthogonalization_4d);
criterion_main!(benches);
