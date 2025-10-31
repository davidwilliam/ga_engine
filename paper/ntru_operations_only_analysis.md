# NTRU N=128: Operations-Only Performance Analysis

## Purpose

This benchmark separates **one-time setup cost** from **repeated operation cost**:

- **Setup** (done once): Convert polynomials to optimal representation
- **Operations** (done many times): Actual multiplication
- **Teardown** (done once): Convert results back

## Why This Matters

In real applications, you typically:
1. **Setup once**: Convert to GA/matrix representation
2. **Operate many times**: Perform hundreds/thousands of multiplications
3. **Teardown once**: Convert final results back

Examples:
- Batch encryption/decryption
- Iterative algorithms
- Polynomial evaluations in loops

---

## Single Operation Results (Setup Excluded)

| Method | Time | vs Karatsuba |
|--------|------|--------------|
| **Karatsuba** | **7.91 µs** | 1.00× (baseline) |
| GA Block (ops only) | 13.12 µs | **0.60× (1.66× slower)** |

**Conclusion:** Even excluding setup, GA block is **1.66× slower** than Karatsuba for single operations.

---

## Batch Operations Results (100 ops, setup excluded)

| Method | Total Time | Time/Op | vs Best | Throughput |
|--------|------------|---------|---------|------------|
| **Toeplitz (classical)** | **342.8 µs** | **3.43 µs** | **1.00×** | 291.75K ops/s |
| Karatsuba | 786.7 µs | 7.87 µs | 0.44× | 127.12K ops/s |
| GA Block (ops only) | 1,318 µs | 13.18 µs | 0.26× | 75.87K ops/s |

**Surprising Result:** Toeplitz matrix-vector multiplication is **FASTEST** for batch operations!

- Toeplitz: **2.29× faster** than Karatsuba
- Toeplitz: **3.84× faster** than GA block
- GA block: **1.68× slower** than Karatsuba

---

## Analysis

### Why Toeplitz Wins for Batch Operations

**Setup Once (excluded from timing):**
```rust
// Convert polynomial to 128×128 Toeplitz matrix
let toeplitz = polynomial_to_toeplitz_matrix_128x128(&a);
```

**Operations (what we measure):**
```rust
for _ in 0..100 {
    result = matrix_vector_multiply(&toeplitz, &b_vec); // O(N²) but fast
}
```

**Advantages:**
1. **Simple loop structure**: No recursion overhead
2. **Excellent cache locality**: Linear memory access
3. **SIMD-friendly**: Compiler auto-vectorization
4. **No allocation**: Pre-allocated result vector
5. **Hardware optimization**: Modern CPUs excel at this

**Complexity:**
- Per operation: O(N²) = O(128²) = 16,384 operations
- But with SIMD, effective operations ≈ 16,384/8 = 2,048 (assuming 8-wide vectors)

### Why Karatsuba is Slower for Batch

**Recursive Structure:**
```rust
fn karatsuba_multiply(a, b) -> result {
    if n <= 8 { return naive_multiply(a, b); }

    // Split
    let (a_low, a_high) = split(a);
    let (b_low, b_high) = split(b);

    // Three recursive calls
    let z0 = karatsuba_multiply(a_low, b_low);    // Recursion!
    let z2 = karatsuba_multiply(a_high, b_high);   // Recursion!
    let z1 = karatsuba_multiply(a_sum, b_sum);     // Recursion!

    // Combine
    return combine(z0, z1, z2);
}
```

**Overhead:**
1. **Recursion**: Function call overhead × log(N) depth
2. **Memory allocation**: Temporary vectors at each level
3. **Split/combine logic**: Additional operations
4. **Less SIMD-friendly**: Irregular access patterns

**Complexity:**
- Per operation: O(N^1.585) = O(128^1.585) ≈ 2,842 operations
- Lower asymptotic complexity, but higher constant factors

### Why GA Block Fails

**Block Decomposition:**
```rust
// 16×16 grid of 8×8 blocks = 256 blocks
for block_row in 0..16 {
    for block_col in 0..16 {
        let mv_result = geometric_product_3d(&mv_block, &b_segment);  // 256 times!
        result[...] += mv_result[...];  // Accumulate
    }
}
```

**Problems:**
1. **256 geometric products**: Each has overhead
2. **Poor cache behavior**: Jumping between many small blocks
3. **Accumulation overhead**: 256 accumulate operations
4. **Float precision**: Loss of accuracy from repeated additions

**Complexity:**
- Per operation: 16² blocks × 8×8 GP ≈ 256 × 64 = 16,384 operations
- Same order as Toeplitz, but with much worse constants

---

## Comparison: All Approaches Including Setup

| Method | Setup Cost | Operation Cost (×100) | Total (100 ops) | Best For |
|--------|------------|----------------------|-----------------|----------|
| Toeplitz | 26.4 µs (matrix) | 342.8 µs | **369.2 µs** | **Batch operations** |
| Karatsuba | 0 µs (none needed) | 786.7 µs | **786.7 µs** | Single operations |
| GA Block | ~1,500 µs (decompose+convert 256 blocks) | 1,318 µs | **~2,818 µs** | Never |

**Break-even analysis:**

**Toeplitz vs Karatsuba:**
- Toeplitz overhead: 26.4 µs
- Toeplitz advantage per op: 7.87 - 3.43 = 4.44 µs
- Break-even: 26.4 / 4.44 ≈ **6 operations**

**For ≥6 operations, Toeplitz is faster than Karatsuba!**

**GA Block:**
- Massive setup overhead (~1,500 µs for 256 block conversions)
- Slower operations (13.18 µs vs 3.43 µs Toeplitz)
- **Never competitive**

---

## Lessons Learned

### ✅ What Works

1. **Toeplitz for batch**: If you can afford one-time setup, Toeplitz dominates
2. **Karatsuba for single ops**: Best for one-off multiplications
3. **Measuring operations separately**: Reveals true computational cost

### ❌ What Doesn't Work

1. **GA block decomposition**: Overhead dominates at all scales
2. **Hierarchical GA**: Block structure destroys locality benefits
3. **Generic assumptions**: "GA is always better" is false

### 🎯 Key Insights

1. **Setup/operation separation is honest**: We're measuring what matters
2. **Asymptotic complexity ≠ practical performance**: Toeplitz O(N²) beats Karatsuba O(N^1.585)
3. **Memory hierarchy matters**: Cache locality > algorithm complexity
4. **SIMD is powerful**: Modern hardware excels at simple loops

---

## Recommendations for Paper

### Accurate Claims

✅ "When operations can be amortized over many multiplications, classical Toeplitz matrix-vector multiplication achieves best performance (2.29× faster than Karatsuba for N=128)"

✅ "For single operations with no setup, Karatsuba remains optimal for N≥64"

✅ "Block-based GA decomposition incurs prohibitive overhead even when excluding conversion costs (1.66× slower than Karatsuba for pure operations)"

✅ "Break-even point between Toeplitz and Karatsuba is ~6 operations, after which Toeplitz dominates"

### Transparent Methodology

We clearly separate:
- **One-time costs** (setup/teardown)
- **Repeated costs** (operations)
- **Real-world use cases** (batch vs single operations)

This is **honest benchmarking** that reflects actual application scenarios.

---

## Conclusion

The operations-only benchmark reveals:

1. **Toeplitz is fastest for batch** (2.29× better than Karatsuba)
2. **GA block fails even without setup** (1.66× worse than Karatsuba)
3. **Setup amortization changes the game** (break-even at ~6 ops)
4. **Memory hierarchy > asymptotic complexity** for practical sizes

**For the paper:** We should present both scenarios:
- **Single operation**: Karatsuba wins (N≥64)
- **Batch operation**: Toeplitz wins (N≥6 ops)
- **GA**: Optimal only for N≤32 (direct mapping, no blocks)

This provides a complete, honest picture of performance trade-offs across different use cases.
