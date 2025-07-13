use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use ga_engine::nd::multivector::Multivector;
use rand::Rng;

/// Analysis of why GA might not help with lattice problems
/// 
/// The key insight is that lattice problems are fundamentally about
/// COMBINATORIAL SEARCH over exponentially large spaces, not about
/// geometric operations themselves.
/// 
/// LLL bottlenecks:
/// 1. Gram-Schmidt orthogonalization (O(n³) per iteration)
/// 2. Size reduction (O(n²) per iteration)  
/// 3. Lovász condition checking (O(n) per iteration)
/// 4. Number of iterations (exponential in worst case)
/// 
/// GA operations are still O(n³) for n-dimensional vectors because:
/// - Multivector space is 2^n dimensional
/// - Inner products still require O(n) operations
/// - Memory overhead is exponential: O(2^n) vs O(n)

fn generate_lattice_basis<const N: usize>(size: usize) -> Vec<VecN<N>> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let coords = std::array::from_fn(|_| rng.gen_range(-100.0..100.0));
            VecN::new(coords)
        })
        .collect()
}

/// Benchmark individual LLL components to identify bottlenecks
fn bench_lll_components_4d(c: &mut Criterion) {
    let basis = generate_lattice_basis::<4>(4);
    
    // Classical Gram-Schmidt
    c.bench_function("gram_schmidt_classical_4d", |b| {
        b.iter(|| {
            let mut orthogonal = Vec::new();
            for (i, v) in basis.iter().enumerate() {
                let mut ortho = v.clone();
                for j in 0..i {
                    let proj_coeff = v.dot(&orthogonal[j]) / orthogonal[j].dot(&orthogonal[j]);
                    let proj = orthogonal[j].clone().scale(proj_coeff);
                    ortho = ortho - proj;
                }
                orthogonal.push(ortho);
            }
            black_box(orthogonal);
        })
    });
    
    // GA-based Gram-Schmidt
    c.bench_function("gram_schmidt_ga_4d", |b| {
        b.iter(|| {
            let mvectors: Vec<_> = basis
                .iter()
                .map(|v| {
                    let mut data = vec![0.0; 1 << 4];
                    for i in 0..4 {
                        data[1 << i] = v.data[i];
                    }
                    Multivector::<4>::new(data)
                })
                .collect();
            
            let mut orthogonal: Vec<Multivector<4>> = Vec::new();
            for (i, mv) in mvectors.iter().enumerate() {
                let mut ortho = mv.clone();
                for j in 0..i {
                    // GA inner product
                    let mut dot = 0.0;
                    for k in 0..4 {
                        dot += mv.data[1 << k] * orthogonal[j].data[1 << k];
                    }
                    
                    let mut norm_sq = 0.0;
                    for k in 0..4 {
                        norm_sq += orthogonal[j].data[1 << k] * orthogonal[j].data[1 << k];
                    }
                    
                    let proj_coeff = dot / norm_sq;
                    
                    // GA scalar multiplication and subtraction
                    let mut proj = vec![0.0; 1 << 4];
                    let mut result = vec![0.0; 1 << 4];
                    for k in 0..4 {
                        proj[1 << k] = orthogonal[j].data[1 << k] * proj_coeff;
                        result[1 << k] = ortho.data[1 << k] - proj[1 << k];
                    }
                    ortho = Multivector::new(result);
                }
                orthogonal.push(ortho);
            }
            black_box(orthogonal);
        })
    });
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let basis = generate_lattice_basis::<8>(8);
    
    c.bench_function("memory_classical_8d", |b| {
        b.iter(|| {
            // Classical: O(n) memory per vector
            let mut working_set = Vec::new();
            for v in &basis {
                working_set.push(v.clone()); // 8 * 8 = 64 bytes
            }
            black_box(working_set);
        })
    });
    
    c.bench_function("memory_ga_8d", |b| {
        b.iter(|| {
            // GA: O(2^n) memory per vector
            let mut working_set = Vec::new();
            for v in &basis {
                let mut data = vec![0.0; 1 << 8]; // 256 * 8 = 2048 bytes
                for i in 0..8 {
                    data[1 << i] = v.data[i];
                }
                working_set.push(Multivector::<8>::new(data));
            }
            black_box(working_set);
        })
    });
}

/// Benchmark the core geometric operations
fn bench_geometric_operations(c: &mut Criterion) {
    let v1 = VecN::<8>::new(std::array::from_fn(|_| rand::thread_rng().gen_range(-10.0..10.0)));
    let v2 = VecN::<8>::new(std::array::from_fn(|_| rand::thread_rng().gen_range(-10.0..10.0)));
    
    c.bench_function("dot_product_classical_8d", |b| {
        b.iter(|| {
            let result = v1.dot(&v2);
            black_box(result);
        })
    });
    
    c.bench_function("dot_product_ga_8d", |b| {
        b.iter(|| {
            let mut mv1_data = vec![0.0; 1 << 8];
            let mut mv2_data = vec![0.0; 1 << 8];
            for i in 0..8 {
                mv1_data[1 << i] = v1.data[i];
                mv2_data[1 << i] = v2.data[i];
            }
            
            let mut result = 0.0;
            for i in 0..8 {
                result += mv1_data[1 << i] * mv2_data[1 << i];
            }
            black_box(result);
        })
    });
    
    c.bench_function("vector_subtraction_classical_8d", |b| {
        b.iter(|| {
            let result = v1.clone() - v2.clone();
            black_box(result);
        })
    });
    
    c.bench_function("vector_subtraction_ga_8d", |b| {
        b.iter(|| {
            let mut mv1_data = vec![0.0; 1 << 8];
            let mut mv2_data = vec![0.0; 1 << 8];
            let mut result_data = vec![0.0; 1 << 8];
            
            for i in 0..8 {
                mv1_data[1 << i] = v1.data[i];
                mv2_data[1 << i] = v2.data[i];
                result_data[1 << i] = mv1_data[1 << i] - mv2_data[1 << i];
            }
            
            let result = Multivector::<8>::new(result_data);
            black_box(result);
        })
    });
}

/// Benchmark scaling with dimension
fn bench_dimensional_scaling(c: &mut Criterion) {
    // 4D comparison
    let basis_4d = generate_lattice_basis::<4>(4);
    c.bench_function("lll_iteration_classical_4d", |b| {
        b.iter(|| {
            let mut working = basis_4d.clone();
            // Single Gram-Schmidt step
            for i in 1..working.len() {
                for j in 0..i {
                    let proj_coeff = working[i].dot(&working[j]) / working[j].dot(&working[j]);
                    let proj = working[j].clone().scale(proj_coeff);
                    working[i] = working[i].clone() - proj;
                }
            }
            black_box(working);
        })
    });
    
    // 8D comparison
    let basis_8d = generate_lattice_basis::<8>(8);
    c.bench_function("lll_iteration_classical_8d", |b| {
        b.iter(|| {
            let mut working = basis_8d.clone();
            // Single Gram-Schmidt step
            for i in 1..working.len() {
                for j in 0..i {
                    let proj_coeff = working[i].dot(&working[j]) / working[j].dot(&working[j]);
                    let proj = working[j].clone().scale(proj_coeff);
                    working[i] = working[i].clone() - proj;
                }
            }
            black_box(working);
        })
    });
    
    // 16D comparison
    let basis_16d = generate_lattice_basis::<16>(16);
    c.bench_function("lll_iteration_classical_16d", |b| {
        b.iter(|| {
            let mut working = basis_16d.clone();
            // Single Gram-Schmidt step
            for i in 1..working.len() {
                for j in 0..i {
                    let proj_coeff = working[i].dot(&working[j]) / working[j].dot(&working[j]);
                    let proj = working[j].clone().scale(proj_coeff);
                    working[i] = working[i].clone() - proj;
                }
            }
            black_box(working);
        })
    });
}

criterion_group!(
    benches,
    bench_lll_components_4d,
    bench_memory_patterns,
    bench_geometric_operations,
    bench_dimensional_scaling
);
criterion_main!(benches); 