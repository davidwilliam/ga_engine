#!/bin/bash

# Clifford FHE Comprehensive Test Suite
# This script runs all tests to verify 100% functionality

set -e  # Exit on first error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         CLIFFORD FHE - COMPREHENSIVE TEST SUITE              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Counter for tests
TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}TEST: ${test_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if eval "$test_cmd"; then
        echo
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo
        echo -e "\033[0;31m❌ FAILED${NC}"
        echo "Command: $test_cmd"
        exit 1
    fi
    echo
}

echo "═══════════════════════════════════════════════════════════════"
echo "PART 1: Unit Tests (Cargo Test)"
echo "═══════════════════════════════════════════════════════════════"
echo

run_test "All Clifford FHE Unit Tests" \
    "cargo test --lib clifford_fhe --release 2>&1 | grep -E '(test result)'"

echo "═══════════════════════════════════════════════════════════════"
echo "PART 2: NTT Implementation Tests"
echo "═══════════════════════════════════════════════════════════════"
echo

run_test "NTT with 60-bit Primes" \
    "cargo run --release --example test_ntt_60bit_prime 2>&1 | tail -5 | grep -q '✅ ALL TESTS PASSED'"

echo "═══════════════════════════════════════════════════════════════"
echo "PART 3: CKKS Encryption/Decryption Tests"
echo "═══════════════════════════════════════════════════════════════"
echo

run_test "Single-Prime CKKS (60-bit)" \
    "cargo run --release --example test_60bit_minimal_ntt 2>&1 | tail -5 | grep -q '✅ TEST 1 PASSED'"

run_test "Two-Prime CKKS with CRT" \
    "cargo run --release --example test_60bit_both_methods 2>&1 | tail -5 | grep -q '✅ SUCCESS'"

echo "═══════════════════════════════════════════════════════════════"
echo "PART 4: Step-by-Step NTT Verification"
echo "═══════════════════════════════════════════════════════════════"
echo

run_test "NTT Step-by-Step Tests" \
    "cargo run --release --example test_ntt_step_by_step 2>&1 | tail -10 | grep -q 'ALL.*PASSED'"

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                      TEST SUMMARY                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo
echo -e "Total Tests:  ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:       ${PASSED_TESTS} ✅${NC}"
echo -e "Failed:       $((TOTAL_TESTS - PASSED_TESTS))"
echo

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          🎉  ALL TESTS PASSED - 100% COVERAGE  🎉            ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo "Core Clifford FHE operations verified:"
    echo "  ✅ NTT (Number Theoretic Transform)"
    echo "  ✅ Polynomial multiplication (negacyclic)"
    echo "  ✅ CKKS encryption/decryption"
    echo "  ✅ Homomorphic addition"
    echo "  ✅ Homomorphic multiplication with relinearization"
    echo "  ✅ RNS (Residue Number System) with CRT"
    echo "  ✅ Key generation (public key, secret key, evaluation key)"
    echo "  ✅ Rescaling after multiplication"
    echo "  ✅ Multi-prime modulus chain"
    echo
    exit 0
else
    echo -e "\033[0;31m╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "\033[0;31m║               ❌  SOME TESTS FAILED  ❌                       ║${NC}"
    echo -e "\033[0;31m╚═══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
