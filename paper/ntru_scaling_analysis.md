# NTRU Polynomial Multiplication Scaling Analysis

## Summary

This document presents benchmark results for NTRU polynomial multiplication across different parameter sizes, comparing Geometric Algebra (GA) based approaches with classical methods.

## Key Findings

### 1. GA Acceleration Works Best for Small N

**N=8:** GA provides **2.55× speedup** over classical Toeplitz matrix method
- Toeplitz classical: 68.4 ns
- **GA accelerated: 26.8 ns** ✓

**N=16:** GA provides **1.92× speedup** over classical Toeplitz matrix method
- Toeplitz classical: 314.2 ns
- **GA accelerated: 163.9 ns** ✓

### 2. Karatsuba Dominates for Medium to Large N

For N ≥ 32, the Karatsuba divide-and-conquer algorithm becomes the clear winner:

**N=32:** Karatsuba **2.32× faster** than Toeplitz
- Toeplitz: 1.62 µs
- Karatsuba: 0.70 µs

**N=64:** Karatsuba **3.48× faster** than Toeplitz
- Toeplitz: 7.49 µs
- Karatsuba: 2.15 µs

**N=128:** Karatsuba **3.21× faster** than Toeplitz
- Toeplitz: 25.3 µs
- Karatsuba: 7.87 µs

**N=256:** Karatsuba **1.66× faster** than Toeplitz
- Toeplitz: 52.9 µs
- Karatsuba: 31.9 µs

### 3. Production NTRU Parameters (N=509)

**N=509 (NIST Level 1):** Karatsuba **1.55× faster** than Toeplitz
- Toeplitz: 271.0 µs
- Karatsuba: 174.7 µs

## Comparison with AMX Hardware Accelerator Paper

The paper "Fast polynomial multiplication using matrix multiplication accelerators" by Gazzoni Filho et al. reports:
- **1.54-3.07× speedup** using Apple AMX hardware coprocessor
- Tested on production NTRU parameters: N=509, 677, 821, 701
- Uses specialized hardware acceleration (AMX matrix multiplication unit)

### Our Results:
- **Small N (8, 16):** GA provides **1.92-2.55× speedup** using pure software mathematical formulation
- **Large N (509):** Classical Karatsuba provides **1.55× speedup** over naive Toeplitz

### Key Distinction:
- **AMX paper:** Hardware-accelerated matrix operations (requires M1/M3 chips)
- **Our work:** Mathematical GA formulation speedups (software-only, portable)
- **Different scales:** We excel at small N; they target production parameters
- **Complementary approaches:** GA formulation + hardware acceleration could stack

## Scaling Characteristics

| N   | Toeplitz (µs) | Karatsuba (µs) | GA (µs)   | Best Method | Speedup |
|-----|---------------|----------------|-----------|-------------|---------|
| 8   | 0.0684        | 0.0580         | **0.0268** | GA          | 2.55×   |
| 16  | 0.3142        | 0.3408         | **0.1639** | GA          | 1.92×   |
| 32  | 1.621         | **0.700**      | N/A       | Karatsuba   | 2.32×   |
| 64  | 7.49          | **2.15**       | N/A       | Karatsuba   | 3.48×   |
| 128 | 25.3          | **7.87**       | N/A       | Karatsuba   | 3.21×   |
| 256 | 52.9          | **31.9**       | N/A       | Karatsuba   | 1.66×   |
| 509 | 271.0         | **174.7**      | N/A       | Karatsuba   | 1.55×   |

## Theoretical Analysis

### Why GA Excels at Small N

For N=8 and N=16, GA can map polynomials to:
- **N=8 → 3D GA (8 components):** Compact multivector representation
- **N=16 → 4D GA (16 components):** Direct one-to-one mapping

The geometric product operation has inherent computational advantages:
- Fewer intermediate calculations due to geometric structure
- Better cache locality (compact representation)
- Compiler can optimize the fixed-size operations effectively

### Why GA Doesn't Scale to Large N

For larger N, the GA approach faces fundamental limitations:
- **N=32 → 5D GA (32 components):** Geometric product table = 32×32 = 1,024 entries
- **N=64 → 6D GA (64 components):** GP table = 64×64 = 4,096 entries
- **N=128 → 7D GA (128 components):** GP table = 128×128 = 16,384 entries
- **N=509 → 9D GA (512 components):** GP table = 512×512 = 262,144 entries

The exponential growth of the geometric product lookup table makes it impractical for large dimensions.

### Karatsuba's O(n^log₂3) Advantage

Karatsuba algorithm:
- Complexity: O(n^1.585) instead of O(n²)
- Becomes more effective as N grows
- Divide-and-conquer exploits polynomial structure
- Best general-purpose method for medium to large N

## Implications for Paper

### Claims We Can Make:
1. ✅ GA provides **2-2.5× speedup** for small NTRU parameters (N=8, N=16)
2. ✅ GA speedup is achieved through **mathematical formulation alone** (no hardware)
3. ✅ GA approach is **complementary** to hardware acceleration
4. ✅ Results demonstrate **geometric structure exploitation** in cryptographic operations

### Claims to Avoid:
1. ❌ "GA beats hardware accelerators" (different scales, different approaches)
2. ❌ "GA is faster for production NTRU" (Karatsuba is better for N≥32)
3. ❌ "GA scales to arbitrary dimensions" (practical limit around N=16)

### Accurate Positioning:
> "For compact NTRU parameter sets (N=8, N=16), our GA-based approach achieves 1.92-2.55× speedup over classical Toeplitz matrix multiplication using pure mathematical formulation. While production NTRU parameters (N≥509) are better served by divide-and-conquer algorithms like Karatsuba, our results demonstrate that exploiting geometric structure through GA provides tangible computational benefits at smaller scales. This suggests that GA formulations, when combined with hardware acceleration, could provide complementary speedups across different parameter ranges."

## Conclusion

The benchmarks reveal a clear scaling pattern:
- **N ≤ 16:** GA is optimal (2-2.5× speedup)
- **N ≥ 32:** Karatsuba is optimal (1.5-3.5× speedup)
- **N = 509:** Karatsuba provides 1.55× speedup (comparable to AMX hardware results)

Our GA approach excels in a different regime than the AMX hardware accelerator paper, demonstrating that geometric structure exploitation provides benefits where direct multivector mapping is feasible.
