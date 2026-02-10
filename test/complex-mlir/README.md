# ComplexF64 MLIR Lowering MWEs (B.6.7)

Investigation: `cursor-toolchain/rules/domains/differentiability/investigations/fft-complex-mlir.md`

## The Two Issues

Both issues block **differentiation** (compilation with `raise=true`) through any model
that uses an FFT-based or Fourier-tridiagonal pressure solver (NonhydrostaticModel,
AnelasticDynamics). Forward-only compilation (without `raise=true`) suppresses the errors.

### Issue 1: `affine.store` type mismatch (Float64 → ComplexF64)

**Error:** `'affine.store' op value to store must have the same type as memref element type`

**Source:** `_compute_source_term!` in `solve_for_pressure.jl:17` and
`_compute_anelastic_source_term!` in `anelastic_pressure_solver.jl:103`

**What happens:** A `Float64` value (velocity divergence) is stored into a `ComplexF64`
array (`solver.source_term` or `solver.storage`). MLIR's `affine.store` requires exact
type match. Reactant doesn't lower Julia's implicit `convert(ComplexF64, Float64)` into
MLIR's `complex.create` operation.

**Fix level:** Local (Oceananigans) — use `Complex(val, zero(val))` in the kernel.

### Issue 2: `llvm.extractvalue` on non-aggregate type

**Error:** `'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f64>'`

**Source:** `copy_real_component!` in `fft_based_poisson_solver.jl:135` and
`fourier_tridiagonal_poisson_solver.jl:257`

**What happens:** `real(ϕc[i, j, k])` extracts the real part of a `ComplexF64`. Julia
lowers `real(z::Complex)` to struct field access → `llvm.extractvalue`. But MLIR
represents complex as native `complex<f64>`, not an LLVM aggregate struct `{f64, f64}`.
`llvm.extractvalue` only works on LLVM aggregates.

**Fix level:** Upstream (Reactant) — use MLIR's `complex.re` dialect op instead.

## File Hierarchy

```
complex-mlir/
├── README.md                              ← This file
│
├── ## Pure MWEs (Reactant + KA only, no Oceananigans)
├── store_real_into_complex_array.jl       ← Issue 1 MWE
├── store_real_into_complex_array_fix.jl   ← Issue 1 proposed fix
├── extract_real_from_complex_array.jl     ← Issue 2 MWE
│
└── ## MedWE (Oceananigans NonhydrostaticModel)
    medwe_nonhydrostatic_fft_derivative.jl ← Both issues via time_step! + raise=true
```

## Running

```bash
# Pure MWEs (fast, no Oceananigans):
julia --project=. test/complex-mlir/store_real_into_complex_array.jl
julia --project=. test/complex-mlir/extract_real_from_complex_array.jl

# MedWE (needs Oceananigans + Enzyme):
julia --project=. test/complex-mlir/medwe_nonhydrostatic_fft_derivative.jl
```
