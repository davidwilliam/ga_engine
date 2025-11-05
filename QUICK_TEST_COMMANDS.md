# Quick Test Commands - Phase 3

## 🎯 Single Command: Verify Everything

```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_phase3_complete
```

**Expected:** 4/4 tests pass with ✅ symbols

**Duration:** ~15 seconds

---

## 🚀 Individual Tests (Copy-Paste Ready)

### Test 1: Basic Rotation
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_rotation_verify
```

### Test 2: Multiple Rotations (k=1,2,4)
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_rotation_multiple
```

### Test 3: Dense Pattern
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_rotation_dense
```

### Test 4: CoeffToSlot/SlotToCoeff
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_coeff_to_slot
```

---

## 🔄 Run All Tests in One Command

```bash
for test in test_phase3_complete test_rotation_verify test_rotation_multiple test_rotation_dense test_coeff_to_slot; do
  echo "━━━ Running $test ━━━"
  RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example $test 2>&1 | grep -E "(✅|❌|PASS|FAIL)" | head -5
  echo ""
done
```

---

## 📊 Performance Timing

```bash
time RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_coeff_to_slot
```

---

## 🏗️ Build Commands

### Clean build
```bash
cargo clean && cargo build --release --features v3
```

### Build all examples
```bash
cargo build --release --features v3 --examples
```

---

## 📝 Documentation

### Generate and open docs
```bash
cargo doc --features v3 --no-deps --open
```

### View specific module
```bash
cargo doc --features v3 --no-deps && open target/doc/ga_engine/clifford_fhe_v3/bootstrapping/index.html
```

---

## ✅ Success Indicators

Look for these in test output:

- `✅ TEST 1 PASSED`
- `✅ TEST 2 PASSED`
- `✅ TEST 3 PASSED`
- `✅ TEST 4 PASSED`
- `✅ ALL TESTS PASSED - PHASE 3: 100% COMPLETE`
- `🎉 Phase 3 Achievement: 100% or Nothing - ACHIEVED!`

---

## ❌ Failure Indicators

If you see these, something is wrong:

- `❌ FAILED`
- `Error:`
- `panicked at`
- Test counts that aren't 4/4

---

## 🎬 Demo Commands

### For quick demo (30 sec):
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_rotation_verify
```

### For presentation (2 min):
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --features v3 --example test_phase3_complete
```

---

## 🔍 Troubleshooting

### If compilation fails:
```bash
cargo clean
rustup update
cargo build --release --features v3
```

### If tests are slow:
Make sure you're using `--release` and `RUSTFLAGS='-C target-cpu=native'`

### If feature v3 not found:
Check `Cargo.toml` has:
```toml
[features]
v3 = []
```
