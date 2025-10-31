# Sparse Matrix Hypothesis: Analysis and Findings

## Hypothesis

**User's Proposal**: "A vector can be converted into a sparse 8×8 matrix (vector in first column, zeros elsewhere) and then transformed into a 3D multivector using our homomorphic mapping. This should allow us to use the 12.4× matrix multiplication speedup for matrix-vector operations."

## Investigation

### What We Thought We Had

We believed we had a homomorphic mapping such that:
```
matrix_8x8_to_multivector3d(A) ⊗ matrix_8x8_to_multivector3d(B)
≈ matrix_8x8_to_multivector3d(A × B)
```

Where ⊗ is the geometric product, and this gives us **12.4× speedup for 128×128 matrix multiplication**.

### Critical Discovery

Upon closer inspection of `benches/matrix_mult_block_ga.rs`, we found that:

**The "matrix multiplication" only computes DIAGONAL elements of the result!**

```rust
// From matrix_mult_block_ga.rs:125
if local_i == local_j && local_i < 8 {
    result[global_i * 128 + global_j] = block_sum[local_i];
}
```

**The homomorphic mapping preserves:**
- ✅ Diagonal elements
- ✅ Certain symmetric/antisymmetric properties
- ❌ **NOT** the full off-diagonal structure

### Why This Matters

#### Matrix × Matrix (Diagonal Only)
- Input: Two 128×128 matrices
- Output: **Diagonal of result** (128 elements)
- GA speedup: **12.4×** ✓

#### Matrix × Vector (All Elements)
- Input: 128×128 matrix and 128×1 vector
- Output: **Full result vector** (128 elements, all matter!)
- GA speedup: **Does not apply** ✗

### Why Sparse Matrix Technique Fails

When we convert a vector to a sparse 8×8 matrix:
```
v = [v₀, v₁, v₂, v₃, v₄, v₅, v₆, v₇]ᵀ

Sparse = [v₀  0  0  0  0  0  0  0]
         [v₁  0  0  0  0  0  0  0]
         [v₂  0  0  0  0  0  0  0]
         [v₃  0  0  0  0  0  0  0]
         [v₄  0  0  0  0  0  0  0]
         [v₅  0  0  0  0  0  0  0]
         [v₆  0  0  0  0  0  0  0]
         [v₇  0  0  0  0  0  0  0]
```

The `matrix_8x8_to_multivector3d` mapping extracts:
- Scalar: (diagonal sum) / 4 → mostly 0
- Vector components: specific diagonal elements → mostly 0
- Bivector components: antisymmetric parts → all 0
- Pseudoscalar: last diagonal → 0

**Result**: Most information about the vector is lost in the mapping!

### Empirical Results

Test case: 128×128 identity matrix × vector of all 1s = vector of all 1s × 128 = [128, 128, 128, ...]

```
Classical result: [32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
GA result:        [4.0, 3.0, 1.0, -1.0, -4.0, -4.0, 0.0, 1.0]
```

**Correctness: FAIL** ❌

Performance:
- Classical: 11.5 µs
- Block GA (sparse): 12.9 µs
- Speedup: **0.89× (slower!)** ❌

## Theoretical Analysis

### The Fundamental Issue

The homomorphic mapping `matrix_8x8_to_multivector3d` was designed for **dense matrices** where:
- Diagonal elements capture trace-like information
- Antisymmetric parts capture rotation-like information
- The geometric product combines these features

For **sparse matrices** (vectors as columns):
- Most extracted features are zero
- The column structure is not preserved
- The geometric product operates on incorrect features

### What Would Be Needed

To make the sparse matrix technique work, we would need a **different homomorphic mapping** that:

1. **Preserves column structure**: Extracts features from columns, not diagonals
2. **Works for sparse matrices**: Handles mostly-zero matrices correctly
3. **Maintains homomorphism**: `f(A) ⊗ f(v_sparse) = f(A × v_sparse)`

Such a mapping would need to be:
- Specifically designed for matrix-vector operations
- Different from our current matrix-matrix mapping
- Proven to preserve the geometric product structure

## Conclusion

**The sparse matrix hypothesis does not work with our current homomorphic mapping.**

### Why the 12.4× Matrix Speedup Doesn't Apply

1. **Limited scope**: Only computes diagonal of result, not full matrix
2. **Dense matrix assumption**: Mapping designed for dense matrices
3. **Different operation**: Matrix × vector needs all elements, not just diagonal

### What We Actually Have

- ✅ 2.58× speedup for polynomial multiplication (N ≤ 32)
- ✅ Diagonal-only matrix multiplication (12.4× for 128×128)
- ❌ Full matrix multiplication (not implemented)
- ❌ Matrix-vector via sparse matrix (doesn't preserve structure)

### Next Steps

To accelerate matrix-vector operations with GA, we need:

1. **Option A**: Design a new homomorphic mapping specifically for vectors
   - Map vectors → multivectors (not via sparse matrices)
   - Ensure geometric product represents matrix-vector multiplication
   - Prove homomorphism property

2. **Option B**: Use different GA structures
   - Outer product for matrix-vector (not geometric product)
   - Sandwich products (A v A†) for transformations
   - Projection operators for subspace operations

3. **Option C**: Accept the limitations
   - GA wins for N ≤ 32 (direct multivector representation)
   - Classical/Karatsuba wins for N > 32
   - Focus paper on the proven speedups, not hypothetical ones

## Lessons Learned

1. **Verify benchmark assumptions**: The "matrix multiplication" was actually diagonal-only
2. **Understand mappings**: Homomorphic mappings have specific domains and codomains
3. **Sparse ≠ Dense**: Techniques for dense matrices don't automatically work for sparse ones
4. **Theory matters**: Can't just "convert to multivector" without understanding the structure

## Status

**Hypothesis: REJECTED** ❌

The sparse matrix technique does not work with our current homomorphic mapping. We need a fundamentally different approach to accelerate matrix-vector operations with GA.
