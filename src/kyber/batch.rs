//! Classical batch encryption (baseline for GA comparison)

use super::classical::{PublicKey, Ciphertext, create_test_message, kyber_encrypt_single};
use super::params::KyberParams;
use rand::Rng;

/// Batch encrypt 4 messages classically (Kyber-512)
///
/// This is the **baseline** for comparison with GA-accelerated batch encryption.
///
/// Classical approach:
/// - Encrypt each message independently
/// - 4 separate matrix-vector operations (each 2×2)
/// - Total: 4 independent encryptions
///
/// Time complexity: 4 × T(single encryption)
pub fn kyber_encrypt_batch_classical_4(
    pk: &PublicKey,
    messages: &[super::polynomial::KyberPoly; 4],
    rng: &mut impl Rng,
) -> [Ciphertext; 4] {
    [
        kyber_encrypt_single(pk, &messages[0], rng),
        kyber_encrypt_single(pk, &messages[1], rng),
        kyber_encrypt_single(pk, &messages[2], rng),
        kyber_encrypt_single(pk, &messages[3], rng),
    ]
}

/// Batch encrypt N messages classically
///
/// Generic version that handles any batch size.
pub fn kyber_encrypt_batch_classical(
    pk: &PublicKey,
    messages: &[super::polynomial::KyberPoly],
    rng: &mut impl Rng,
) -> Vec<Ciphertext> {
    messages
        .iter()
        .map(|msg| kyber_encrypt_single(pk, msg, rng))
        .collect()
}

/// Generate batch of test messages for benchmarking
pub fn create_test_messages_batch(
    params: KyberParams,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<super::polynomial::KyberPoly> {
    (0..count)
        .map(|_| create_test_message(params, rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::classical::kyber_keygen;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_batch_encrypt_4() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let (pk, _sk) = kyber_keygen(params, &mut rng);

        // Create 4 messages
        let messages = [
            create_test_message(params, &mut rng),
            create_test_message(params, &mut rng),
            create_test_message(params, &mut rng),
            create_test_message(params, &mut rng),
        ];

        // Batch encrypt
        let ciphertexts = kyber_encrypt_batch_classical_4(&pk, &messages, &mut rng);

        // Verify all ciphertexts were created
        assert_eq!(ciphertexts.len(), 4);
        for ct in &ciphertexts {
            assert_eq!(ct.u.polys.len(), params.k);
            assert_eq!(ct.v.coeffs.len(), params.n);
        }
    }

    #[test]
    fn test_batch_encrypt_generic() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let (pk, _sk) = kyber_keygen(params, &mut rng);

        // Create batch of messages
        let messages = create_test_messages_batch(params, 10, &mut rng);

        // Batch encrypt
        let ciphertexts = kyber_encrypt_batch_classical(&pk, &messages, &mut rng);

        assert_eq!(ciphertexts.len(), 10);
    }
}
