#=
Investigation: affine.if inside affine.parallel cannot be raised (B.6.7 remaining blocker)
Status:       FAILING (as of 2026-02-17)
Purpose:      Pure KA + Reactant MWE reproducing "cannot raise if yet" for
              affine.if inside affine.parallel.

Background
----------
After working around the two ComplexF64 MLIR bugs in B.6.7 (by routing through
Float64 scratch arrays and broadcasts), the resulting all-Float64 KA kernels still
fail with `raise=true`. The `raise-affine-to-stablehlo` pass cannot convert an
`affine.if` containing non-pure operations (affine.load / affine.store) when it is
nested inside `affine.parallel`.

Root cause
----------
Oceananigans `heuristic_workgroup(Nx, Ny)` always returns `(16, 16)` for any
`Nx, Ny ≥ 2`. For a 4×4 grid this means:
    workgroup = (16, 16) = 256 threads  >  ndrange = (4, 4) = 16 valid threads
KA pads the iteration to fill the full workgroup and emits an `affine.if` guard:

    affine.parallel (%arg3) = (0) to (256) {
      affine.if affine_set<(d0) : (d0 mod 16 + 1 >= 0, -(d0 mod 16) + 3 >= 0,
                                   d0 floordiv 16 + 1 >= 0, -(d0 floordiv 16) + 3 >= 0)>(%arg3) {
        %0 = affine.load ...      ← non-pure (memory read)
        ...
        affine.store ...          ← non-pure (memory write)
      }
    }

Error:    "cannot raise if yet (non-pure or yielded values)"
Pass:      raise-affine-to-stablehlo (triggered by Reactant.@compile raise=true)

Observed in
-----------
test/test_reactant_complex_kernels.jl — both FFTBasedPoissonSolver and
FourierTridiagonalPoissonSolver testsets hit this on compute_source_term!.
The call chain is:
    compute_source_term!(solver, nothing, velocities, Δt)
      └─ launch!(arch, grid, :xyz, _compute_source_term!, scratch, grid, Ũ)
           └─ _compute_source_term! kernel with workgroup(16,16) on ndrange(4,4)
                └─ affine.if generated → raise fails

This MWE isolates the pattern to pure KA + Reactant (no Oceananigans required).

Related
-------
cursor-toolchain/rules/domains/differentiability/investigations/fft-complex-mlir.md
  → "Remaining Blocker: Generic failed to raise func"
test/test_reactant_complex_kernels.jl  (Oceananigans-level reproduction)

Run
---
    TEST_FILE=raise-affine-if/mwe_affine_if_parallel.jl julia --project test/runtests.jl
=#

using Test
using Reactant
using KernelAbstractions: @kernel, @index
using CUDA

Reactant.set_default_backend("cpu")

# Enable MLIR dumping so the generated IR can be inspected offline.
mlir_dump_dir = joinpath(@__DIR__, "mlir_dump")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir
@info "MLIR dumps will be written to: $(mlir_dump_dir)"

# Obtain the Reactant KernelAbstractions backend explicitly.
# Inside @compile, arrays become TracedRArray and we need ReactantBackend().
const _ReactantKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const ReactantBackend = _ReactantKAExt.ReactantBackend

#####
##### Grid parameters (mirror the failing test: 4×4 grid, halo=3)
#####

const Nx, Ny = 4, 4
const HALO   = 3
const WG     = (16, 16)  # Oceananigans heuristic_workgroup(4, 4) → (16, 16)

#####
##### Kernel 1: Minimal — plain identity-like read/write, no index offsets.
#####
# Tests whether affine.if alone (not the halo arithmetic) is the blocker.
# MLIR body inside affine.if: affine.load + arith.addf + affine.store — all non-pure.
#####

@kernel function _minimal_write!(out, inp)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = inp[i, j] + 1.0
end

#####
##### Kernel 2: Halo-offset divergence — mirrors _compute_source_term! in
#####           Oceananigans.Models.NonhydrostaticModels.
#####
# Reads from (Nx+2H)×(Ny+2H) halo-padded velocity arrays with interior offset +HALO.
# Writes du/dx + dv/dy to a Nx×Ny output array.
# This is the exact pattern that fails in FFTBasedPoissonSolver and
# FourierTridiagonalPoissonSolver after the B.6.7.1/B.6.7.2 ComplexF64 workarounds.
#####

@kernel function _divergence_like!(out, u, v, scale)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Finite-difference divergence: (u[i+1,j] - u[i,j])/Δx + (v[i,j+1] - v[i,j])/Δy
        # Interior indices start at HALO+1 = 4 in the padded arrays.
        du = u[i + HALO + 1, j + HALO] - u[i + HALO, j + HALO]
        dv = v[i + HALO, j + HALO + 1] - v[i + HALO, j + HALO]
        out[i, j] = (du + dv) * scale
    end
end

#####
##### Launcher helpers
#####
# workgroup WG=(16,16) on ndrange (Nx,Ny)=(4,4):
#   16×16 = 256 total threads  >  4×4 = 16 valid → affine.if guard emitted by KA.

function run_minimal!(out, inp)
    _minimal_write!(ReactantBackend(), WG)(out, inp; ndrange=(Nx, Ny))
    return nothing
end

function run_divergence!(out, u, v, scale)
    _divergence_like!(ReactantBackend(), WG)(out, u, v, scale; ndrange=(Nx, Ny))
    return nothing
end

#####
##### Tests
#####

@testset "affine.if in affine.parallel: raise-affine-to-stablehlo limitation (B.6.7)" begin
    # ── Array setup ──────────────────────────────────────────────────────────
    inp_min  = Reactant.to_rarray(rand(Float64, Nx, Ny))
    out_min  = Reactant.to_rarray(zeros(Float64, Nx, Ny))

    H = HALO
    u_pad    = Reactant.to_rarray(rand(Float64, Nx + 2H, Ny + 2H))  # 10×10
    v_pad    = Reactant.to_rarray(rand(Float64, Nx + 2H, Ny + 2H))  # 10×10
    out_div  = Reactant.to_rarray(zeros(Float64, Nx, Ny))
    scale    = 4.0  # 1/Δx for a uniform grid with Δx = 1/4

    # ── Without raise=true ───────────────────────────────────────────────────
    # The raise-affine-to-stablehlo pass is NOT run; compilation succeeds.
    # These tests establish that the kernels are otherwise correct.

    @testset "raise=false: minimal kernel compiles and produces finite output" begin
        @info "  Compiling minimal kernel (raise=false)..."
        compiled! = Reactant.@compile sync=true run_minimal!(out_min, inp_min)
        @test compiled! !== nothing
        compiled!(out_min, inp_min)
        result = Array(out_min)
        @test all(isfinite, result)
        @test all(result .> 0)   # inp values are positive rand() + 1.0
    end

    @testset "raise=false: divergence-like kernel compiles and produces finite output" begin
        @info "  Compiling divergence-like kernel (raise=false)..."
        compiled! = Reactant.@compile sync=true run_divergence!(out_div, u_pad, v_pad, scale)
        @test compiled! !== nothing
        compiled!(out_div, u_pad, v_pad, scale)
        result = Array(out_div)
        @test all(isfinite, result)
    end

    # ── With raise=true ──────────────────────────────────────────────────────
    # Expected to fail: "cannot raise if yet (non-pure or yielded values)"
    # from the raise-affine-to-stablehlo pass.
    # Marked @test_broken because the upstream Reactant fix is pending.

    @testset "raise=true: minimal kernel (expected: affine.if raise failure)" begin
        @info "  Compiling minimal kernel (raise=true) — expected to fail..."
        @test_broken begin
            compiled! = Reactant.@compile raise_first=true raise=true sync=true run_minimal!(out_min, inp_min)
            compiled!(out_min, inp_min)
            true
        end
    end

    @testset "raise=true: divergence-like kernel (expected: affine.if raise failure)" begin
        @info "  Compiling divergence-like kernel (raise=true) — expected to fail..."
        @test_broken begin
            compiled! = Reactant.@compile raise_first=true raise=true sync=true run_divergence!(out_div, u_pad, v_pad, scale)
            compiled!(out_div, u_pad, v_pad, scale)
            true
        end
    end
end
