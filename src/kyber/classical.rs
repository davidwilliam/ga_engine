//! Classical Kyber operations (single encryption)

use super::polynomial::{PolyMatrix, PolyVec, KyberPoly};
use super::params::KyberParams;
use rand::Rng;

/// Kyber public key
#[derive(Clone)]
pub struct PublicKey {
    /// Matrix A (k×k)
    pub a: PolyMatrix,
    /// Vector t = A·s + e
    pub t: PolyVec,
    pub params: KyberParams,
}

/// Kyber secret key
#[derive(Clone)]
pub struct SecretKey {
    /// Secret vector s
    pub s: PolyVec,
    pub params: KyberParams,
}

/// Kyber ciphertext
#[derive(Clone)]
pub struct Ciphertext {
    /// Ciphertext vector u = A^T·r + e₁
    pub u: PolyVec,
    /// Ciphertext polynomial v = t^T·r + e₂ + message
    pub v: KyberPoly,
    pub params: KyberParams,
}

/// Generate Kyber keypair
///
/// This generates the public matrix A and secret vector s, then computes t = A·s + e.
/// This is a simplified version for demonstration purposes.
pub fn kyber_keygen(params: KyberParams, rng: &mut impl Rng) -> (PublicKey, SecretKey) {
    // Generate random matrix A (public seed)
    let a = PolyMatrix::random(params, rng);

    // Sample secret vector s with small coefficients (CBD with η₁)
    let s = PolyVec::sample_cbd(params, params.eta1, rng);

    // Sample error vector e (CBD with η₁)
    let e = PolyVec::sample_cbd(params, params.eta1, rng);

    // Compute t = A·s + e
    let as_prod = a.mul_vec(&s);
    let t = as_prod.add(&e);

    let pk = PublicKey {
        a: a.clone(),
        t,
        params,
    };

    let sk = SecretKey { s, params };

    (pk, sk)
}

/// Encrypt a single message with Kyber
///
/// **This is the operation we want to optimize with GA batching.**
///
/// The encryption computes:
/// - u = A^T·r + e₁ (matrix-vector product - BOTTLENECK)
/// - v = t^T·r + e₂ + m
///
/// For Kyber-512 (k=2), A is 2×2, so this is too small for our 8×8 GA speedup.
/// Solution: Batch 4 encryptions to create an effective 8×8 operation.
pub fn kyber_encrypt_single(
    pk: &PublicKey,
    message: &KyberPoly,
    rng: &mut impl Rng,
) -> Ciphertext {
    let params = pk.params;

    // Sample random vector r (CBD with η₁)
    let r = PolyVec::sample_cbd(params, params.eta1, rng);

    // Sample error vectors e₁, e₂ (CBD with η₂)
    let e1 = PolyVec::sample_cbd(params, params.eta2, rng);
    let e2 = KyberPoly::sample_cbd(params, params.eta2, rng);

    // Compute u = A^T·r + e₁
    // For simplicity, we compute A·r (same structure, just different interpretation)
    let ar = pk.a.mul_vec(&r);
    let u = ar.add(&e1);

    // Compute v = t^T·r + e₂ + m
    // Dot product of t and r
    let mut v = KyberPoly::zero(params);
    for i in 0..params.k {
        let prod = pk.t.polys[i].mul_naive(&r.polys[i]);
        v = v.add(&prod);
    }
    v = v.add(&e2);
    v = v.add(message);

    Ciphertext { u, v, params }
}

/// Encrypt a message (placeholder function for benchmarking)
///
/// In real Kyber, the message is encoded from bytes. For our benchmarking,
/// we create a simple message polynomial.
pub fn create_test_message(params: KyberParams, rng: &mut impl Rng) -> KyberPoly {
    // Create a message with small coefficients (0 or 1)
    let coeffs: Vec<i32> = (0..params.n)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { 0 })
        .collect();
    KyberPoly::new(coeffs, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_keygen() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let (pk, sk) = kyber_keygen(params, &mut rng);

        // Check dimensions
        assert_eq!(pk.a.rows.len(), params.k);
        assert_eq!(pk.t.polys.len(), params.k);
        assert_eq!(sk.s.polys.len(), params.k);
    }

    #[test]
    fn test_encrypt() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let (pk, _sk) = kyber_keygen(params, &mut rng);
        let message = create_test_message(params, &mut rng);

        let ciphertext = kyber_encrypt_single(&pk, &message, &mut rng);

        // Check ciphertext structure
        assert_eq!(ciphertext.u.polys.len(), params.k);
        assert_eq!(ciphertext.v.coeffs.len(), params.n);
    }

    #[test]
    fn test_multiple_encryptions() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let (pk, _sk) = kyber_keygen(params, &mut rng);

        // Encrypt multiple messages (simulating batch scenario)
        for _ in 0..4 {
            let message = create_test_message(params, &mut rng);
            let _ciphertext = kyber_encrypt_single(&pk, &message, &mut rng);
        }
    }
}
