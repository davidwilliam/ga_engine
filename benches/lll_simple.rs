use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use ga_engine::nd::multivector::Multivector;
use rand::Rng;

const DELTA: f64 = 0.75; // Lovász parameter

/// Generate a random lattice basis that needs reduction
fn generate_bad_basis() -> Vec<VecN<4>> {
    let mut rng = rand::thread_rng();
    let mut basis = Vec::new();
    
    // Create a basis with large, nearly dependent vectors
    for i in 0..4 {
        let mut coords = [0.0; 4];
        for j in 0..4 {
            coords[j] = if i == j { 
                rng.gen_range(100.0..1000.0) // Large diagonal entries
            } else {
                rng.gen_range(-10.0..10.0) // Small off-diagonal entries
            };
        }
        basis.push(VecN::new(coords));
    }
    basis
}

/// Classical LLL step
fn classical_lll_step(basis: &mut Vec<VecN<4>>) {
    let n = basis.len();
    let mut k = 1;
    
    while k < n {
        // Gram-Schmidt orthogonalization
        let mut gram_schmidt = Vec::new();
        let mut mu = vec![vec![0.0; n]; n];
        
        for i in 0..=k {
            gram_schmidt.push(basis[i].clone());
            
            for j in 0..i {
                mu[i][j] = basis[i].dot(&gram_schmidt[j]) / gram_schmidt[j].dot(&gram_schmidt[j]);
                let proj = gram_schmidt[j].clone().scale(mu[i][j]);
                gram_schmidt[i] = gram_schmidt[i].clone() - proj;
            }
        }
        
        // Size reduction
        for l in (0..k).rev() {
            if mu[k][l].abs() > 0.5 {
                let q = mu[k][l].round();
                let scaled = basis[l].clone().scale(q);
                basis[k] = basis[k].clone() - scaled;
                
                // Update mu coefficients
                for j in 0..l {
                    mu[k][j] -= q * mu[l][j];
                }
                mu[k][l] -= q;
            }
        }
        
        // Check Lovász condition
        if k > 0 {
            let left = gram_schmidt[k].dot(&gram_schmidt[k]);
            let right = (DELTA - mu[k][k-1].powi(2)) * gram_schmidt[k-1].dot(&gram_schmidt[k-1]);
            
            if left < right {
                // Swap basis vectors
                basis.swap(k, k-1);
                k = k.saturating_sub(1).max(1);
            } else {
                k += 1;
            }
        } else {
            k += 1;
        }
    }
}

/// GA-based LLL step
fn ga_lll_step(basis: &mut Vec<VecN<4>>) {
    // Convert to multivectors
    let mut mvectors: Vec<_> = basis
        .iter()
        .map(|v| {
            let mut data = vec![0.0; 16]; // 2^4 = 16
            for i in 0..4 {
                data[1 << i] = v.data[i];
            }
            Multivector::<4>::new(data)
        })
        .collect();
    
    let n = mvectors.len();
    let mut k = 1;
    
    while k < n {
        // GA-based Gram-Schmidt orthogonalization
        let mut gram_schmidt = Vec::new();
        let mut mu = vec![vec![0.0; n]; n];
        
        for i in 0..=k {
            gram_schmidt.push(mvectors[i].clone());
            
            for j in 0..i {
                // GA inner product
                let mut dot = 0.0;
                let mut norm_sq = 0.0;
                for l in 0..4 {
                    dot += mvectors[i].data[1 << l] * gram_schmidt[j].data[1 << l];
                    norm_sq += gram_schmidt[j].data[1 << l] * gram_schmidt[j].data[1 << l];
                }
                
                mu[i][j] = dot / norm_sq;
                
                // GA scalar multiplication and subtraction
                let mut proj = vec![0.0; 16];
                let mut result = vec![0.0; 16];
                for l in 0..4 {
                    proj[1 << l] = gram_schmidt[j].data[1 << l] * mu[i][j];
                    result[1 << l] = gram_schmidt[i].data[1 << l] - proj[1 << l];
                }
                gram_schmidt[i] = Multivector::new(result);
            }
        }
        
        // Size reduction
        for l in (0..k).rev() {
            if mu[k][l].abs() > 0.5 {
                let q = mu[k][l].round();
                
                // GA scalar multiplication and subtraction
                let mut scaled = vec![0.0; 16];
                let mut result = vec![0.0; 16];
                for m in 0..4 {
                    scaled[1 << m] = mvectors[l].data[1 << m] * q;
                    result[1 << m] = mvectors[k].data[1 << m] - scaled[1 << m];
                }
                mvectors[k] = Multivector::new(result);
                
                for j in 0..l {
                    mu[k][j] -= q * mu[l][j];
                }
                mu[k][l] -= q;
            }
        }
        
        // Check Lovász condition
        if k > 0 {
            let mut left = 0.0;
            let mut right_base = 0.0;
            for l in 0..4 {
                left += gram_schmidt[k].data[1 << l] * gram_schmidt[k].data[1 << l];
                right_base += gram_schmidt[k-1].data[1 << l] * gram_schmidt[k-1].data[1 << l];
            }
            
            let right = (DELTA - mu[k][k-1].powi(2)) * right_base;
            
            if left < right {
                mvectors.swap(k, k-1);
                k = k.saturating_sub(1).max(1);
            } else {
                k += 1;
            }
        } else {
            k += 1;
        }
    }
    
    // Convert back to basis vectors
    for (i, mv) in mvectors.iter().enumerate() {
        for j in 0..4 {
            basis[i].data[j] = mv.data[1 << j];
        }
    }
}

fn bench_lll_4d(c: &mut Criterion) {
    c.bench_function("lll_classical_4d", |b| {
        b.iter(|| {
            let mut basis = generate_bad_basis();
            classical_lll_step(black_box(&mut basis));
            black_box(basis);
        })
    });
    
    c.bench_function("lll_ga_4d", |b| {
        b.iter(|| {
            let mut basis = generate_bad_basis();
            ga_lll_step(black_box(&mut basis));
            black_box(basis);
        })
    });
}

fn bench_gram_schmidt_4d(c: &mut Criterion) {
    let basis = generate_bad_basis();
    
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
    
    c.bench_function("gram_schmidt_ga_4d", |b| {
        b.iter(|| {
            let mvectors: Vec<_> = basis
                .iter()
                .map(|v| {
                    let mut data = vec![0.0; 16];
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
                    let mut dot = 0.0;
                    let mut norm_sq = 0.0;
                    for k in 0..4 {
                        dot += mv.data[1 << k] * orthogonal[j].data[1 << k];
                        norm_sq += orthogonal[j].data[1 << k] * orthogonal[j].data[1 << k];
                    }
                    
                    let proj_coeff = dot / norm_sq;
                    
                    let mut proj = vec![0.0; 16];
                    let mut result = vec![0.0; 16];
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

criterion_group!(benches, bench_lll_4d, bench_gram_schmidt_4d);
criterion_main!(benches); 