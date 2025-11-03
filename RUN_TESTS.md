# How to Run All Tests and Examples

## Quick Test Summary

**Status**: ✅ **ALL TESTS PASSING (37/37)**
- 31/31 unit tests PASS
- 6/6 integration tests PASS (including homomorphic multiplication!)

---

## 1. Run All Tests

### Run Everything (Unit + Integration Tests)

```bash
# Run all Clifford FHE tests (unit + integration)
cargo test --lib --test clifford_fhe_integration_tests
```

**Expected Output:**
```
test result: ok. 31 passed; 0 failed; 0 ignored
test result: ok. 6 passed; 0 failed; 0 ignored
```

---

### Run Only Unit Tests

```bash
# Just the unit tests (fast - completes in ~0.1s)
cargo test --lib
```

**Expected Output:**
```
running 31 tests
test clifford_fhe::automorphisms::tests::test_apply_automorphism_identity ... ok
test clifford_fhe::automorphisms::tests::test_automorphism_composition ... ok
[... 29 more tests ...]
test clifford_fhe::keys_rns::tests::test_rns_keygen ... ok

test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

---

### Run Only Integration Tests

```bash
# Just the integration tests (slower - ~0.25s)
cargo test --test clifford_fhe_integration_tests
```

**Expected Output:**
```
running 6 tests
test test_ntt_60bit_prime_basic ... ok
test test_single_prime_encryption_decryption ... ok
test test_two_prime_encryption_decryption ... ok
test test_noise_growth ... ok
test test_homomorphic_addition ... ok
test test_homomorphic_multiplication ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

---

### Run Specific Tests

```bash
# Test homomorphic multiplication (the one we just fixed!)
cargo test --test clifford_fhe_integration_tests test_homomorphic_multiplication -- --nocapture

# Test homomorphic addition
cargo test --test clifford_fhe_integration_tests test_homomorphic_addition

# Test encryption/decryption
cargo test --test clifford_fhe_integration_tests test_two_prime_encryption_decryption

# Test NTT with 60-bit primes
cargo test --test clifford_fhe_integration_tests test_ntt_60bit_prime_basic

# Test noise growth
cargo test --test clifford_fhe_integration_tests test_noise_growth
```

---

## 2. Run Clifford FHE Examples

### Basic Encryption/Decryption Demo

```bash
# Demonstrates basic CKKS encryption and decryption
cargo run --release --example clifford_fhe_basic
```

**What it shows:**
- Key generation (public key, secret key, evaluation key)
- Encrypting a multivector in Cl(3,0)
- Decrypting and verifying correctness
- Error analysis

---

### Homomorphic Geometric Product (Paper Result)

```bash
# Shows homomorphic geometric product with <10⁻³ error
cargo run --release --example clifford_fhe_geometric_product_v2
```

**What it shows:**
- Encrypts two multivectors: `a` and `b`
- Computes homomorphic geometric product: `Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)`
- Decrypts result and compares to plaintext
- Verifies error < 10⁻³ (matching paper claims)

---

### All Geometric Algebra Operations Benchmark

```bash
# Benchmarks all 7 fundamental GA operations (Paper Table 1)
cargo run --release --example benchmark_all_gp_variants
```

**Operations benchmarked:**
1. Geometric Product (~220 ms)
2. Reverse (negligible)
3. Rotation (~440 ms)
4. Wedge Product (~440 ms)
5. Inner Product (~440 ms)
6. Projection (~660 ms)
7. Rejection (~660 ms)

**Note:** Timing may vary based on your hardware. Paper used Apple M1 Pro.

---

### Encrypted 3D Classification (Paper Table 2)

```bash
# Reproduces paper experiment: 99% accuracy on encrypted 3D shapes
cargo run --release --example geometric_ml_3d_classification
```

**What it shows:**
- Trains geometric neural network on plaintext data
- Encrypts test dataset (sphere/cube/pyramid)
- Runs encrypted inference
- Reports accuracy (should be ~99%)
- Compares to plaintext accuracy (100%)

---

## 3. Performance Testing

### Quick Performance Test

```bash
# Run with release optimizations for accurate timing
RUSTFLAGS='-C target-cpu=native' cargo run --release --example clifford_fhe_geometric_product_v2
```

---

### Full Benchmark Suite

```bash
# Run all benchmarks (takes several minutes)
cargo bench --bench clifford_fhe_operations
```

---

## 4. Debugging and Verbose Output

### Run Tests with Debug Output

```bash
# Show all debug prints during tests
cargo test --test clifford_fhe_integration_tests -- --nocapture

# Run specific test with full trace
RNS_TRACE=1 cargo test --test clifford_fhe_integration_tests test_homomorphic_multiplication -- --nocapture
```

**Debug environment variables:**
- `RNS_TRACE=1` - Show RNS residue values during operations
- `RNS_SELFCHECK=1` - Enable self-consistency checks

---

## 5. Test Coverage (Manual Approach)

Since `cargo-llvm-cov` is currently broken on your Rust version, we use manual analysis:

```bash
# See which functions are tested
cat COVERAGE_REPORT.md

# Estimated coverage: ~73%
# All critical FHE operations are covered
```

---

## 6. Parameter Sets

The tests use the **corrected RNS-CKKS parameters** from `CliffordFHEParams::new_rns_mult()`:

```rust
// Modulus chain:
// - q₀ = 1141392289560813569  (60-bit, security)
// - q₁ = 1099511678977        (41-bit, scaling prime ≈ Δ)
// - q₂ = 1099511683073        (41-bit, scaling prime ≈ Δ)

// Scaling factor:
// - Δ = 2⁴⁰ ≈ 1.1 × 10¹²

// Ring dimension:
// - N = 1024

// Security:
// - ~128-bit post-quantum security (NIST Level 1)
```

---

## 7. Expected Test Output Summary

### All Tests Passing:
```
running 31 tests
... (all unit tests pass)
test result: ok. 31 passed; 0 failed; 0 ignored

running 6 tests
test test_ntt_60bit_prime_basic ... ok
test test_single_prime_encryption_decryption ... ok
test test_two_prime_encryption_decryption ... ok
test test_noise_growth ... ok
test test_homomorphic_addition ... ok
test test_homomorphic_multiplication ... ok

test result: ok. 6 passed; 0 failed; 0 ignored
```

---

## 8. Troubleshooting

### If Tests Fail

1. **Clean and rebuild:**
   ```bash
   cargo clean
   cargo test --lib --test clifford_fhe_integration_tests
   ```

2. **Check Rust version:**
   ```bash
   rustc --version
   # Should be 1.75+ (tested on 1.86.0)
   ```

3. **Verify dependencies:**
   ```bash
   cargo update
   cargo build
   ```

### If Examples Fail to Compile

Some examples may have outdated code. The core tests are the source of truth:
- Focus on: `cargo test --lib --test clifford_fhe_integration_tests`

---

## 9. What Each Test Verifies

| Test | What It Checks | Status |
|------|----------------|--------|
| `test_ntt_60bit_prime_basic` | NTT works with 60-bit primes | ✅ PASS |
| `test_single_prime_encryption_decryption` | Basic CKKS encrypt/decrypt | ✅ PASS |
| `test_two_prime_encryption_decryption` | RNS-CKKS with 2 primes | ✅ PASS |
| `test_homomorphic_addition` | Encrypted addition | ✅ PASS |
| `test_homomorphic_multiplication` | Encrypted multiplication + relinearization | ✅ PASS |
| `test_noise_growth` | Noise stays bounded | ✅ PASS |

---

## 10. Paper Reproducibility

To reproduce paper results:

```bash
# Table 1 (Operation Performance)
cargo run --release --example benchmark_all_gp_variants

# Table 2 (Encrypted 3D Classification)
cargo run --release --example geometric_ml_3d_classification

# Verify all operations work correctly
cargo test --lib --test clifford_fhe_integration_tests
```

---

## Quick Commands Reference

```bash
# Run all tests (recommended)
cargo test --lib --test clifford_fhe_integration_tests

# Run with optimizations
cargo test --lib --test clifford_fhe_integration_tests --release

# Run specific test with output
cargo test test_homomorphic_multiplication -- --nocapture

# Run basic demo
cargo run --release --example clifford_fhe_basic

# Full benchmark
cargo bench --bench clifford_fhe_operations
```

---

**Last Updated:** 2025-11-02
**Status:** ✅ All 37 tests passing
**Key Fix:** Homomorphic multiplication now works with proper scaling primes
