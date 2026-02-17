# Investigation: `affine.if` in `affine.parallel` — raise-affine-to-stablehlo failure

**Related Known Issue:** B.6.7 ComplexF64 MLIR Lowering — "Remaining Blocker: Generic failed to raise func"  
**Investigation File:** `cursor-toolchain/rules/domains/differentiability/investigations/fft-complex-mlir.md`  
**Status:** FAILING (as of 2026-02-17)  
**Packages Affected:** Oceananigans (NonhydrostaticModel), Breeze (AnelasticDynamics)  

---

## Summary

After the B.6.7.1 and B.6.7.2 ComplexF64 MLIR workarounds were applied in
`OceananigansReactantExt/Solvers.jl`, the resulting all-Float64 KA kernels still fail
when compiled with `raise=true`. The `raise-affine-to-stablehlo` pass cannot lower
`affine.if` blocks that contain non-pure operations (`affine.load` / `affine.store`)
when nested inside an `affine.parallel` loop.

This is the **current frontier blocker** for FFT-based pressure solver differentiation.

---

## The MLIR Pattern

For a 4×4 grid, `heuristic_workgroup(4, 4)` → `(16, 16)`. Since the workgroup
(256 threads) exceeds the ndrange (16 valid threads), KernelAbstractions emits an
`affine.if` bounds guard:

```mlir
affine.parallel (%arg3) = (0) to (256) {
  affine.if affine_set<(d0) : (d0 mod 16 + 1 >= 0, -(d0 mod 16) + 3 >= 0,
                               d0 floordiv 16 + 1 >= 0, -(d0 floordiv 16) + 3 >= 0)>(%arg3) {
    %0 = affine.load %arg1[...]    ← non-pure
    ...
    affine.store %n, %arg0[...]    ← non-pure
  }
}
```

The `raise-affine-to-stablehlo` pass then fails with:
```
error: cannot raise if yet (non-pure or yielded values)
```

---

## Grid-Size Sensitivity

The simple `affine.if` raise failure (compute_source_term!) is **weirdly specific to certain
grid sizes**. It appears when `workgroup > ndrange` which triggers the `affine.if` bounds guard.
This can often be worked around by adjusting the grid size so that `ndrange >= workgroup` (e.g.,
using `size=(16, 16)` instead of `size=(4, 4)`). This issue is being **worked on upstream** in Reactant.

---

## Two Sub-Issues

### 1. Simple kernel + affine.if (compute_source_term!)

`compute_source_term!` uses a simple element-wise kernel dispatched over `:xyz`.
When workgroup > ndrange, the `affine.if` guard with `affine.load`/`affine.store`
cannot be raised. **Grid-size-dependent; can often be worked around.**

MWE: `mwe_tiny.jl`

### 2. Tridiagonal kernel + affine.if + scf.while (solve!)

`solve_batched_tridiagonal_system_kernel!` dispatches 2D `(i, j)` with ndrange `(5, 5)`,
workgroup `(16, 16)`. The body contains sequential Thomas-algorithm `for` loops (forward +
backward sweeps) that lower to `scf.while` nested inside the `affine.if`. This is potentially
harder to raise even if the simple affine.if is fixed, since the `scf.while` loops interact
with the affine context.

MWE: `mwe_tridiagonal_loop.jl`

---

## Files

| File | Purpose |
|------|---------|
| `mwe_tiny.jl` | Minimal MWE — simple write, workgroup > ndrange |
| `mwe_affine_if_parallel.jl` | Broader MWE with divergence-like kernel + halo offsets |
| `mwe_tridiagonal_loop.jl` | MWE for Thomas-algorithm loops inside affine.if |
| `mlir_dump/` | MLIR dumps captured during test runs (gitignored) |

---

## How to Run

```bash
cd /path/to/Oceananigans.jl

# Run just the MWE
TEST_FILE=raise-affine-if/mwe_affine_if_parallel.jl julia --project -e 'using Pkg; Pkg.test()'

# Run the full reactant_2 group (includes test_reactant_complex_kernels.jl)
TEST_GROUP=reactant_2 julia --project -e 'using Pkg; Pkg.test()'
```

Expected output for the `raise=true` testsets: `Test Broken` (not `Error` — the
`@test_broken` macro handles the expected exception).

---

## Current Status (2026-02-17)

| Test | raise=false | raise=true | Notes |
|------|-------------|------------|-------|
| Minimal write kernel | ✅ Pass | ❌ `cannot raise if yet` | Grid-size sensitive; upstream WIP |
| Divergence-like kernel (halo offsets) | ✅ Pass | ❌ `cannot raise if yet` | Grid-size sensitive; upstream WIP |
| Thomas-algorithm loop kernel | TBD | ❌ `cannot raise if yet` (expected) | May persist after simple fix |

---

## Path to Resolution

1. **Upstream Reactant fix** — teach `raise-affine-to-stablehlo` to handle
   `affine.if` with non-pure bodies inside `affine.parallel`.  
   Possible strategies:
   - Convert `affine.if` → `scf.if` before running the raise pass.
   - Use predicated execution (`arith.select`) for the guarded body.
   - Expand `affine.parallel` to `scf.for` + `scf.if` as a pre-pass.

2. **Downstream workaround** — if the upstream fix is unavailable, consider
   launching kernels with workgroup = ndrange (no padding) to avoid emitting
   `affine.if` altogether. This would require overriding `heuristic_workgroup`
   for Reactant architectures.

---

## Related Issues / PRs

- None filed yet — this MWE is the first step toward an upstream Reactant issue.
- Related: B.6.2 (large grid CPU failure) involves a similar `failed to raise func`
  but from a different code path (OffsetStaticSize).
