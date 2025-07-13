use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::vecn::VecN;
use ga_engine::nd::multivector::Multivector;
use rand::Rng;

const DELTA: f64 = 0.75; // Lovász parameter

/// Generate a random lattice basis that's deliberately "bad" (needs reduction)
fn generate_bad_basis<const N: usize>(size: usize) -> Vec<VecN<N>> {
    let mut rng = rand::thread_rng();
    let mut basis = Vec::new();
    
    // Create a basis with large, nearly dependent vectors
    for i in 0..size {
        let mut coords = [0.0; N];
        for j in 0..N {
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

/// Classical LLL implementation
struct ClassicalLLL<const N: usize> {
    basis: Vec<VecN<N>>,
    gram_schmidt: Vec<VecN<N>>,
    mu: Vec<Vec<f64>>,
}

impl<const N: usize> ClassicalLLL<N> {
    fn new(basis: Vec<VecN<N>>) -> Self {
        let size = basis.len();
        Self {
            basis,
            gram_schmidt: vec![VecN::new([0.0; N]); size],
            mu: vec![vec![0.0; size]; size],
        }
    }
    
    /// Gram-Schmidt orthogonalization step
    fn gram_schmidt_step(&mut self, k: usize) {
        self.gram_schmidt[k] = self.basis[k].clone();
        
        for j in 0..k {
            self.mu[k][j] = self.basis[k].dot(&self.gram_schmidt[j]) / self.gram_schmidt[j].dot(&self.gram_schmidt[j]);
            let proj = self.gram_schmidt[j].clone().scale(self.mu[k][j]);
            self.gram_schmidt[k] = self.gram_schmidt[k].clone() - proj;
        }
    }
    
    /// Size reduction step
    fn size_reduce(&mut self, k: usize, l: usize) {
        if self.mu[k][l].abs() > 0.5 {
            let q = self.mu[k][l].round();
            let scaled = self.basis[l].clone().scale(q);
            self.basis[k] = self.basis[k].clone() - scaled;
            
            // Update Gram-Schmidt coefficients
            for j in 0..l {
                self.mu[k][j] -= q * self.mu[l][j];
            }
            self.mu[k][l] -= q;
        }
    }
    
    /// Check Lovász condition
    fn lovasz_condition(&self, k: usize) -> bool {
        let left = self.gram_schmidt[k].dot(&self.gram_schmidt[k]);
        let right = (DELTA - self.mu[k][k-1].powi(2)) * self.gram_schmidt[k-1].dot(&self.gram_schmidt[k-1]);
        left >= right
    }
    
    /// Single LLL iteration
    fn lll_step(&mut self) -> bool {
        let n = self.basis.len();
        let mut k = 1;
        
        while k < n {
            // Gram-Schmidt orthogonalization
            self.gram_schmidt_step(k);
            
            // Size reduction
            for l in (0..k).rev() {
                self.size_reduce(k, l);
            }
            
            // Check Lovász condition
            if k > 0 && !self.lovasz_condition(k) {
                // Swap basis vectors
                self.basis.swap(k, k-1);
                k = k.saturating_sub(1).max(1);
            } else {
                k += 1;
            }
        }
        true
    }
}

/// GA-based LLL implementation using multivectors
struct GABasedLLL<const N: usize> {
    basis: Vec<Multivector<N>>,
    gram_schmidt: Vec<Multivector<N>>,
    mu: Vec<Vec<f64>>,
}

impl<const N: usize> GABasedLLL<N> {
    fn new(basis: Vec<VecN<N>>) -> Self {
        let mvectors: Vec<_> = basis
            .iter()
            .map(|v| {
                let mut data = vec![0.0; 1 << N];
                for i in 0..N {
                    data[1 << i] = v.data[i];
                }
                Multivector::<N>::new(data)
            })
            .collect();
        
        let size = mvectors.len();
        Self {
            basis: mvectors,
            gram_schmidt: vec![Multivector::new(vec![0.0; 1 << N]); size],
            mu: vec![vec![0.0; size]; size],
        }
    }
    
    /// GA-based inner product
    fn inner_product(&self, a: &Multivector<N>, b: &Multivector<N>) -> f64 {
        // Extract vector components and compute dot product
        let mut sum = 0.0;
        for i in 0..N {
            sum += a.data[1 << i] * b.data[1 << i];
        }
        sum
    }
    
    /// GA-based vector subtraction
    fn subtract(&self, a: &Multivector<N>, b: &Multivector<N>) -> Multivector<N> {
        let mut result = vec![0.0; 1 << N];
        for i in 0..N {
            result[1 << i] = a.data[1 << i] - b.data[1 << i];
        }
        Multivector::new(result)
    }
    
    /// GA-based scalar multiplication
    fn scale(&self, a: &Multivector<N>, scalar: f64) -> Multivector<N> {
        let mut result = vec![0.0; 1 << N];
        for i in 0..N {
            result[1 << i] = a.data[1 << i] * scalar;
        }
        Multivector::new(result)
    }
    
    /// GA-based Gram-Schmidt step
    fn gram_schmidt_step(&mut self, k: usize) {
        self.gram_schmidt[k] = self.basis[k].clone();
        
        for j in 0..k {
            self.mu[k][j] = self.inner_product(&self.basis[k], &self.gram_schmidt[j]) / 
                           self.inner_product(&self.gram_schmidt[j], &self.gram_schmidt[j]);
            let proj = self.scale(&self.gram_schmidt[j], self.mu[k][j]);
            self.gram_schmidt[k] = self.subtract(&self.gram_schmidt[k], &proj);
        }
    }
    
    /// GA-based size reduction
    fn size_reduce(&mut self, k: usize, l: usize) {
        if self.mu[k][l].abs() > 0.5 {
            let q = self.mu[k][l].round();
            let scaled = self.scale(&self.basis[l], q);
            self.basis[k] = self.subtract(&self.basis[k], &scaled);
            
            for j in 0..l {
                self.mu[k][j] -= q * self.mu[l][j];
            }
            self.mu[k][l] -= q;
        }
    }
    
    /// Check Lovász condition (same as classical)
    fn lovasz_condition(&self, k: usize) -> bool {
        let left = self.inner_product(&self.gram_schmidt[k], &self.gram_schmidt[k]);
        let right = (DELTA - self.mu[k][k-1].powi(2)) * 
                   self.inner_product(&self.gram_schmidt[k-1], &self.gram_schmidt[k-1]);
        left >= right
    }
    
    /// Single LLL iteration
    fn lll_step(&mut self) -> bool {
        let n = self.basis.len();
        let mut k = 1;
        
        while k < n {
            self.gram_schmidt_step(k);
            
            for l in (0..k).rev() {
                self.size_reduce(k, l);
            }
            
            if k > 0 && !self.lovasz_condition(k) {
                self.basis.swap(k, k-1);
                k = k.saturating_sub(1).max(1);
            } else {
                k += 1;
            }
        }
        true
    }
}

fn bench_lll_4d(c: &mut Criterion) {
    let basis = generate_bad_basis::<4>(4);
    
    c.bench_function("lll_classical_4d", |b| {
        b.iter(|| {
            let mut lll = ClassicalLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
    
    c.bench_function("lll_ga_4d", |b| {
        b.iter(|| {
            let mut lll = GABasedLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
}

fn bench_lll_8d(c: &mut Criterion) {
    let basis = generate_bad_basis::<8>(8);
    
    c.bench_function("lll_classical_8d", |b| {
        b.iter(|| {
            let mut lll = ClassicalLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
    
    c.bench_function("lll_ga_8d", |b| {
        b.iter(|| {
            let mut lll = GABasedLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
}

fn bench_lll_16d(c: &mut Criterion) {
    let basis = generate_bad_basis::<16>(16);
    
    c.bench_function("lll_classical_16d", |b| {
        b.iter(|| {
            let mut lll = ClassicalLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
    
    c.bench_function("lll_ga_16d", |b| {
        b.iter(|| {
            let mut lll = GABasedLLL::new(black_box(basis.clone()));
            black_box(lll.lll_step());
        })
    });
}

criterion_group!(benches, bench_lll_4d, bench_lll_8d, bench_lll_16d);
criterion_main!(benches); 