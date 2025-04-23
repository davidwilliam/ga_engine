# ROADMAP

We now have a rock-solid GA kernel. To turn that into a true GA Engine and demonstrate real-world wins:

1. Build semantic adapters & canonical use-cases
  1.1 Implement a Rotor3::rotate(v: Vec3) -> Vec3 and compare it to your classical::apply_matrix3(&m, &v)—unit-test equality and bench them head-to-head.
  1.2 Similarly, a GA-based reflection, projection, or change-of-basis.
2. Package a clean Rust API
  2.1 Expose Multivector3, Rotor3, Bivector3, etc., with ergonomic constructors, methods, and doc-comments.
  2.2 Publish on crates.io as ga_engine so any Rust dev can cargo add ga_engine.
3. Implement “killer-apps” in GA
  3.1 Neural layer: dense 8→8, convolution 3×3, PCA/ SVD on small matrices—first classical, then GA sandwich forms.
  3.2 FHE primitives: key-generation or ciphertext rotation expressed in GA, showing a drop-in replacement.
4. Extend benchmarks
  4.1 For each adapter above, write a Criterion bench comparing the same inputs → same outputs, measuring total end-to-end time.
  4.2 Compute GFLOP/s or “GGA-ops/s” and report side-by-side.
5. Document & publish
  5.1 Flesh out the README with these new examples, include code snippets and charts.
  5.2 Provide a tutorial Notebook or small app that highlights “drop in your classical code, switch one import, see a 2× performance win.”

## Next

- Micro-optimize rotate_fast
  - Inline its arithmetic, unroll the cross-product, add #[inline(always)]—you’ll likely bring it below classical speed.
  - Package the API
- Export Vec3, apply_matrix3, Rotor3::{rotate, rotate_fast}, multiply_matrices, geometric_product_full in your crate’s public API.
  - Add comprehensive docs/examples so users can copy-paste.
- Killer-app demos
  - Neural layer: show GA replacing a tiny dense layer (8→8) in an ML pipeline, matching inference results with lower latency.
  - FHE primitive: implement a GA-based ciphertext rotation and bench it against the classical matrix method.
- Publish & visualize
  - Update your README with a bar chart (e.g. via Criterion’s CSV + Python/Matplotlib) showing these five benchmarks side-by-side.
  - Release to crates.io—now any Rust dev can cargo add ga_engine and immediately see the performance story.

