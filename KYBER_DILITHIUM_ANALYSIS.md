# CRYSTALS-Kyber/Dilithium: GA Acceleration Opportunity Analysis

**Date**: October 31, 2025
**Status**: Planning Phase - Post NTRU Success (2.57× speedup achieved)

---

## Executive Summary

After achieving **2.57× speedup on NTRU**, we're ready to tackle **CRYSTALS-Kyber** (NIST-selected encryption) and **CRYSTALS-Dilithium** (NIST-selected signatures).

**Key Insight**: Both algorithms use **matrix-vector polynomial multiplication** as their computational bottleneck - exactly where our GA approach excels.

---

## CRYSTALS-Kyber (Encryption/KEM)

### Overview
- **Status**: NIST-selected standard (2022)
- **Type**: Key Encapsulation Mechanism (KEM)
- **Security**: Module-LWE (Learning With Errors)
- **Ring**: Rq = Zq[x]/(x^256 + 1), q = 3329

### Core Operation: Matrix-Vector Multiplication

**Encryption**: `c = A·s + e` (matrix-vector product)
- A: k×k matrix of polynomials (public key)
- s: k-vector of polynomials (secret)
- e: k-vector of error polynomials

**Key Generation**: `t = A·s + e`
- Same operation structure

### Security Levels & Matrix Sizes

| Security Level | k (matrix dimension) | Equivalent Security | Our GA Opportunity |
|----------------|---------------------|---------------------|-------------------|
| **Kyber-512** | k=2 | AES-128 | ❌ Too small (2×2) |
| **Kyber-768** | k=3 | AES-192 | ⚠️ Marginal (3×3) |
| **Kyber-1024** | k=4 | AES-256 | ⚠️ Marginal (4×4) |

**Problem**: Individual matrix sizes (2×2, 3×3, 4×4) are **too small** for our 8×8 or 16×16 GA speedups.

### ✅ SOLUTION: Batch Processing

**Key insight from research**:
> "Efficient matrix multiplications of sizes **greater than four** can be used to generate/verify **m signatures** with Dilithium and **greater than two** to encrypt **m messages** with Kyber"

**Batch Encryption Strategy**:
```
Instead of:  4 separate 2×2 operations (Kyber-512)
Do:          1 combined 8×8 operation (process 4 encryptions at once)

Expected speedup: 2.57× (same as NTRU N=8)
```

**Real-world relevance**:
- TLS handshakes: batch encrypt session keys
- VPN connections: batch encrypt multiple packets
- Cloud services: batch encrypt multiple requests
- IoT gateways: batch encrypt sensor data

---

## CRYSTALS-Dilithium (Digital Signatures)

### Overview
- **Status**: NIST-selected standard (2022)
- **Type**: Digital Signature Algorithm
- **Security**: Module-LWE
- **Ring**: Rq = Zq[x]/(x^256 + 1), q = 2^23 - 2^13 + 1

### Core Operation: Matrix-Vector Multiplication

**Signing**: `t = A·s₁ + s₂` (matrix-vector product)
- A: k×ℓ matrix of polynomials
- s₁: ℓ-vector of polynomials
- s₂: k-vector of polynomials

### Security Levels & Matrix Sizes

| Security Level | k×ℓ (matrix dims) | Public Key | Signature | Our GA Opportunity |
|----------------|------------------|------------|-----------|-------------------|
| **Dilithium2** | 4×4 | 1312 bytes | 2420 bytes | ⚠️ Marginal (4×4) |
| **Dilithium3** | 6×5 | 1952 bytes | 3293 bytes | ❌ Odd size |
| **Dilithium5** | 8×7 | 2592 bytes | 4595 bytes | ⚠️ Non-square (8×7) |

**Problem**: Matrix sizes don't align well with our 8×8 or 16×16 GA optimizations.

### ✅ SOLUTION: Batch Verification

**Batch Signature Verification**:
```
Instead of:  Verify 8 signatures individually (8× 4×4 operations)
Do:          1 combined verification using larger matrix ops

Expected speedup: 1.5-2× on batch operations
```

**Real-world relevance**:
- Certificate validation: verify multiple signatures in TLS chains
- Software updates: verify signatures on multiple packages
- Blockchain: verify multiple transaction signatures
- Email: verify signatures on multiple messages

---

## Our GA Advantage Strategy

### What Works (Based on NTRU Success)

✅ **N=8 operations**: 2.57× speedup achieved
✅ **N=16 operations**: 1.91× speedup achieved
✅ **Batch processing**: 2.56× speedup on 100 ops

### Kyber/Dilithium Strategy

**Don't try to accelerate individual operations** (matrices too small)

**DO accelerate batch operations**:

1. **Kyber Batch Encryption** (Most Promising)
   - Process 4 Kyber-512 encryptions → single 8×8 operation
   - Process 2 Kyber-1024 encryptions → single 8×8 operation
   - Expected: **2.5× speedup** on batches

2. **Kyber Batch Decryption**
   - Similar approach for server-side batch decryption
   - TLS servers decrypt many session keys

3. **Dilithium Batch Verification**
   - Combine multiple signature verifications
   - Expected: **1.5-2× speedup**

---

## Implementation Plan

### Phase 1: Kyber Batch Encryption (Highest Priority)

**Why start here**:
- ✅ Clear use case (TLS, VPN, cloud services)
- ✅ Direct mapping to 8×8 operations
- ✅ Similar to NTRU (proven 2.57× speedup)
- ✅ NIST-selected standard (highest impact)

**Implementation Steps**:
1. Implement basic Kyber-512 single encryption (classical)
2. Implement Kyber-512 batch encryption (4 at once)
3. Map batch operation to 8×8 GA matrix operations
4. Benchmark: classical vs GA batch processing

**Expected Result**: 2.5× speedup on batches of 4

### Phase 2: Dilithium Batch Verification

**Why second**:
- ✅ Complements Kyber (encryption + signatures)
- ✅ Practical use case (certificate chains, updates)
- ⚠️ More complex than Kyber
- ⚠️ Non-square matrices (harder to optimize)

### Phase 3: Extended Analysis

- Test larger batch sizes (8, 16 encryptions)
- Measure overhead of batch coordination
- Compare with hardware accelerators

---

## Technical Challenges

### Challenge 1: NTT Operations

**Issue**: Kyber/Dilithium use Number Theoretic Transform (NTT) for polynomial multiplication
- NTT is already highly optimized (320 cycles on modern CPUs)
- May be harder to beat than NTRU's simpler convolution

**Strategy**:
- Focus on **batch coordination** overhead
- GA advantage in **combining multiple operations**
- Not trying to beat individual NTT

### Challenge 2: Small Matrix Sizes

**Issue**: Individual operations use 2×2, 3×3, or 4×4 matrices
- Too small for our 8×8 GA optimizations

**Solution**: **Only target batch operations**
- Combine 2-4 operations into single 8×8
- This is the honest, practical approach

### Challenge 3: Polynomial Ring Arithmetic

**Issue**: Operations in Zq[x]/(x^256 + 1) with modular reduction
- More complex than NTRU's simpler ring

**Strategy**:
- Implement correct polynomial arithmetic first
- Then optimize with GA
- Validate against test vectors

---

## Success Criteria

### Minimum Viable Success
- ✅ **1.5× speedup** on Kyber batch encryption (4 messages)
- Claim: "GA provides practical speedups for batched post-quantum operations"

### Target Success
- ✅ **2.0-2.5× speedup** on Kyber batch encryption
- Claim: "GA achieves 2× speedup on NIST-standard Kyber batches"

### Exceptional Success
- ✅ **>2.5× speedup** on Kyber batches
- ✅ **>1.5× speedup** on Dilithium batches
- Claim: "GA accelerates both NIST-selected post-quantum standards"

---

## Comparison: NTRU vs Kyber

| Aspect | NTRU | Kyber |
|--------|------|-------|
| **Status** | Finalist (not selected) | ✅ **NIST Selected** |
| **Matrix Size** | N=8 direct | 2×2 (batch to 8×8) |
| **Our Speedup** | 2.57× | 2.0-2.5× (target) |
| **Use Case** | Academic | **Production standard** |
| **Impact** | Good | **🔥 MAXIMUM** |

**Why Kyber matters more**:
- It's the **official NIST standard**
- Actual deployments are happening **now**
- Industry will adopt it (browsers, OS, cloud)
- **Higher citation/recognition potential**

---

## Timeline Estimate

**Phase 1: Kyber Batch (2-3 days)**
- Day 1: Implement classical Kyber-512 basics
- Day 2: Implement batch encryption + GA mapping
- Day 3: Benchmark and validate

**Phase 2: Dilithium Batch (2-3 days)**
- Day 1: Implement classical Dilithium2 basics
- Day 2: Implement batch verification + GA mapping
- Day 3: Benchmark and validate

**Total**: 4-6 days for both algorithms

---

## Decision: What to Implement?

### Option 1: Kyber Only (Recommended)
**Pros**:
- ✅ Highest impact (NIST-selected encryption)
- ✅ Clearer use case (batch encryption)
- ✅ Simpler than Dilithium
- ✅ Direct comparison to NTRU success

**Cons**:
- ⚠️ Only covers encryption (not signatures)

### Option 2: Both Kyber + Dilithium
**Pros**:
- ✅ Complete NIST coverage (encryption + signatures)
- ✅ Stronger publication claim

**Cons**:
- ⚠️ More time investment
- ⚠️ Dilithium is more complex
- ⚠️ Diminishing returns (one strong result > two weak results)

### Option 3: Skip to FHE (Homomorphic Encryption)
**Pros**:
- ✅ Extreme performance bottleneck (1000-1000000× slower)
- ✅ Any speedup is valuable
- ✅ Hot research area

**Cons**:
- ⚠️ Very complex to implement correctly
- ⚠️ Requires deep FHE knowledge
- ⚠️ Harder to validate correctness

---

## Recommendation

### START WITH: Kyber Batch Encryption

**Rationale**:
1. **Maximum Impact**: NIST-selected standard
2. **Clear Use Case**: Batch encryption is practical
3. **Proven Approach**: Similar to NTRU (2.57× success)
4. **Manageable Scope**: 2-3 days implementation
5. **Publication-Ready**: "GA accelerates NIST Kyber"

**If successful** (2×+ speedup):
- You have encryption (Kyber) + previous crypto (NTRU)
- Strong publication story
- Consider adding Dilithium later

**If marginal** (1.3-1.5× speedup):
- Still publishable (honest result)
- Move to FHE or other applications
- Don't spend more time optimizing

---

## Next Action

**Let's implement Kyber Batch Encryption!**

I'll create:
1. ✅ Basic Kyber-512 polynomial operations
2. ✅ Classical batch encryption (baseline)
3. ✅ GA-accelerated batch encryption
4. ✅ Comprehensive benchmarks
5. ✅ Correctness tests with NIST test vectors

**Target**: Demonstrate **2× speedup** on batches of 4 Kyber-512 encryptions.

**Ready to proceed?**
