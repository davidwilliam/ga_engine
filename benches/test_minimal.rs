use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simple_test(c: &mut Criterion) {
    c.bench_function("simple_addition", |b| {
        b.iter(|| {
            let result = black_box(1 + 1);
            black_box(result)
        })
    });
}

criterion_group!(benches, simple_test);
criterion_main!(benches); 