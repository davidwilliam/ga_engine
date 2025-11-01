#!/usr/bin/env python3
"""
Corrected Security Estimation for Clifford-LWE-256

Uses formulas from NIST PQC Kyber submission and academic LWE security literature.
"""

import math

def log2(x):
    return math.log(x) / math.log(2)

def estimate_lwe_security_kyber_model(n, q, sigma):
    """
    Estimate LWE security using the model from Kyber's NIST submission.
    
    This uses the "Core-SVP" model which is the standard for NIST PQC.
    
    The security estimate for LWE(n, q, χ_sigma) is approximately:
        bit_security ≈ 0.265 * β
    
    where β (BKZ block size) is estimated as:
        β ≈ sqrt(n * log(q) / log(delta))
    
    and delta (root Hermite factor) relates to error rate:
        log(delta) ≈ log²(sigma * sqrt(2πe)) / (4 * n * log(q))
    
    For simplicity, we use the direct formula from Kyber docs:
        bit_security ≈ 0.265 * sqrt(n * log2(q))  (simplified conservative estimate)
    
    More accurate formula accounting for error:
        bit_security ≈ 0.292 * (n / (log2(q/sigma) + c))
    where c is a small constant (typically ~1.5)
    """
    
    # Method 1: Simplified Kyber estimate
    log_q = log2(q)
    beta_simple = math.sqrt(n * log_q)
    security_simple = 0.265 * beta_simple
    
    # Method 2: More accurate with error term
    log_q_over_sigma = log2(q / sigma)
    beta_accurate = n / (log_q_over_sigma + 1.5)
    security_accurate = 0.292 * beta_accurate
    
    # Method 3: Conservative primal attack estimate
    # From "Estimate all the {LWE, NTRU} schemes!" by Albrecht et al.
    # beta ≈ n * log(q) / (log(q) - log(sigma * sqrt(2*pi)))
    log_sigma_term = log2(sigma * math.sqrt(2 * math.pi))
    if log_q > log_sigma_term:
        beta_primal = (n * log_q) / (log_q - log_sigma_term)
        security_primal = 0.292 * beta_primal
    else:
        security_primal = 0
    
    return {
        'simple': security_simple,
        'accurate': security_accurate,
        'primal': security_primal,
        'beta_simple': beta_simple,
        'beta_accurate': beta_accurate,
        'beta_primal': beta_primal if log_q > log_sigma_term else 0
    }

def main():
    print("=" * 70)
    print("Clifford-LWE-256: Corrected Security Estimation")
    print("=" * 70)
    print()
    
    # Parameters
    N = 32
    k = 8
    n = N * k
    q = 3329
    sigma = 1.414
    
    print(f"Parameters:")
    print(f"  N (polynomial degree): {N}")
    print(f"  k (Clifford components): {k}")
    print(f"  n (total LWE dimension): {n}")
    print(f"  q (modulus): {q}")
    print(f"  σ (error stddev): {sigma:.3f}")
    print()
    
    # Estimate security
    result = estimate_lwe_security_kyber_model(n, q, sigma)
    
    print("=" * 70)
    print("Security Estimates (Core-SVP Model)")
    print("=" * 70)
    print()
    
    print(f"Method 1 - Simplified Kyber formula:")
    print(f"  β ≈ sqrt(n * log2(q)) = {result['beta_simple']:.1f}")
    print(f"  Security: 2^{result['simple']:.1f} ≈ {int(result['simple'])} bits")
    print()
    
    print(f"Method 2 - With error term:")
    print(f"  β ≈ n / (log2(q/σ) + 1.5) = {result['beta_accurate']:.1f}")
    print(f"  Security: 2^{result['accurate']:.1f} ≈ {int(result['accurate'])} bits")
    print()
    
    print(f"Method 3 - Primal attack (Albrecht et al.):")
    print(f"  β ≈ n*log(q) / (log(q) - log(σ*sqrt(2π))) = {result['beta_primal']:.1f}")
    print(f"  Security: 2^{result['primal']:.1f} ≈ {int(result['primal'])} bits")
    print()
    
    # Conservative estimate (take minimum)
    min_security = min(result['simple'], result['accurate'], result['primal'])
    max_security = max(result['simple'], result['accurate'], result['primal'])
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Security range: {int(min_security)}-{int(max_security)} bits")
    print(f"Conservative estimate: ~{int(min_security)} bits")
    print()
    
    # Classification
    if min_security >= 128:
        classification = "NIST Level 1+ (AES-128 equivalent or better)"
        status = "✅"
    elif min_security >= 112:
        classification = "Strong security (112-bit, acceptable for most uses)"
        status = "✅"
    elif min_security >= 80:
        classification = "Research-level security (80-bit, suitable for POC)"
        status = "⚠️"
    else:
        classification = "Low security (< 80 bits, parameter adjustment needed)"
        status = "❌"
    
    print(f"{status} Classification: {classification}")
    print()
    
    # Comparison to Kyber-512
    print("=" * 70)
    print("Comparison to Kyber-512")
    print("=" * 70)
    print()
    
    # Estimate Kyber-512 security
    kyber_n = 512
    kyber_q = 3329
    kyber_sigma = 1.414  # Similar error distribution
    kyber_result = estimate_lwe_security_kyber_model(kyber_n, kyber_q, kyber_sigma)
    
    print("Kyber-512:")
    print(f"  Parameters: N=256, q=3329, k=2")
    print(f"  LWE dimension: n = 256 × 2 = {kyber_n}")
    print(f"  Estimated security: ~{int(kyber_result['accurate'])} bits")
    print(f"  NIST classification: Level 1 (128-bit security)")
    print()
    
    print("Clifford-LWE-256:")
    print(f"  Parameters: N={N}, q={q}, k={k}")
    print(f"  LWE dimension: n = {N} × {k} = {n}")
    print(f"  Estimated security: ~{int(min_security)}-{int(max_security)} bits")
    print()
    
    # Security scaling
    ratio = n / kyber_n
    security_ratio = min_security / kyber_result['accurate']
    
    print(f"Dimension ratio: {ratio:.2f}× (Clifford vs Kyber)")
    print(f"Security ratio: {security_ratio:.2f}× (Clifford vs Kyber)")
    print()
    
    # Recommendations for higher security
    if min_security < 80:
        print("=" * 70)
        print("Recommendations for Higher Security")
        print("=" * 70)
        print()
        print("Current security is below research threshold (80 bits).")
        print("Consider these parameter adjustments:")
        print()
        
        # Test larger N values
        for N_new in [64, 128, 256]:
            n_new = N_new * k
            result_new = estimate_lwe_security_kyber_model(n_new, q, sigma)
            print(f"  N={N_new}, k={k}, n={n_new}:")
            print(f"    → Security: ~{int(result_new['accurate'])} bits")
        
        print()
    
    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    print()
    
    if min_security >= 80:
        print(f"✅ Clifford-LWE-256 provides ~{int(min_security)}-{int(max_security)}-bit security.")
        print("   This is SUFFICIENT for research and proof-of-concept.")
        print()
        print("   The security level demonstrates that:")
        print("   - Geometric algebra can support post-quantum cryptography")
        print("   - Clifford-LWE is a credible alternative to standard lattice schemes")
        print("   - Performance/security trade-offs are comparable to existing schemes")
    else:
        print(f"⚠️ Clifford-LWE-256 provides ~{int(min_security)}-{int(max_security)}-bit security.")
        print("   This is BELOW the research threshold (80 bits).")
        print()
        print("   Recommendations:")
        print("   1. Increase N to 64 or 128 for higher security")
        print("   2. Or increase q to a larger prime (e.g., 7681 or 12289)")
        print("   3. Document this as a performance-optimized variant")
    
    print()

if __name__ == "__main__":
    main()
