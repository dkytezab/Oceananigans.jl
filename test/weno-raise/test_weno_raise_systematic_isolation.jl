"""
Systematic isolation of the WENO raise=true failure.

We know:
  - `_biased_interpolate_xᶠᵃᵃ(i,j,k, grid, WENO{3}, LeftBias(), c)` on BPB grid FAILS
  - The same on PPP grid PASSES

This file calls different sub-pieces of that function separately to find
which component is necessary and sufficient for the failure.
"""

using Test
using Random

using Oceananigans
using Reactant

using KernelAbstractions: @kernel, @index

using Oceananigans.Advection:
    WENO, Centered, LeftBias,
    _biased_interpolate_xᶠᵃᵃ,
    biased_interpolate_xᶠᵃᵃ,
    _____biased_interpolate_xᶠᵃᵃ,
    outside_biased_halo_xᶠ,
    weno_stencil_x,
    biased_weno_weights,
    weno_reconstruction

using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, topology, architecture
using Oceananigans.Utils: launch!

# ── Shared grid + field setup (BPB topology only, the failing case) ──

function build_bpb()
    arch = ReactantState()
    grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1),
                           topology=(Bounded, Periodic, Bounded))
    c   = CenterField(grid)
    out = XFaceField(grid)
    Random.seed!(1234)
    set!(c, randn(size(c)...))
    return grid, c, out
end

# ── Probe wrapper ──

function probe(label, run_fn!, out, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(out, args...)
        f!(out, args...)
        @info "  → PASS" label
        return (ok=true, error="")
    catch err
        buf = IOBuffer(); showerror(buf, err)
        msg = String(take!(buf))
        @info "  → FAIL" label
        return (ok=false, error=msg)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: WENO{3} biased_interpolate (NO ifelse wrapping, no bounded path)
#   Calls biased_interpolate_xᶠᵃᵃ directly, bypassing topology dispatch.
#   If this passes → WENO math itself is fine; the bug is in the wrapping.
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_weno3_direct!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c)
end

function run_weno3_direct!(out, grid, scheme, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_weno3_direct!, out, grid, scheme, c)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Full _biased_interpolate_xᶠᵃᵃ with topology dispatch (the failing case)
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_full_topo_dispatch!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c)
end

function run_full_topo_dispatch!(out, grid, scheme, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_full_topo_dispatch!, out, grid, scheme, c)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Manual single-level ifelse wrapping
#   ifelse(halo_check, weno3_direct, centered_direct)
#   Skips the WENO{2} buffer — just one level of ifelse.
#   If this fails → single-level ifelse + WENO math is enough.
#   If this passes → the nested buffer recursion is needed to trigger it.
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_single_ifelse!(out, grid, scheme, buffer, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(
        outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme),
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c),
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, buffer, LeftBias(), c))
end

function run_single_ifelse!(out, grid, scheme, buffer, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_single_ifelse!, out, grid, scheme, buffer, c)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: ifelse wrapping with ONLY simple (Centered) interpolation
#   ifelse(halo_check_outer, centered_interp, centered_interp)
#   No WENO math at all, but keeps the ifelse + Oceananigans field access.
#   If this fails → the issue is in ifelse + field access, not WENO.
#   If this passes → WENO math contributes to the failure.
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_ifelse_centered_only!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    centered = Centered(order=2)
    @inbounds out[i, j, k] = ifelse(
        outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme),
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, centered, LeftBias(), c),
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, centered, LeftBias(), c))
end

function run_ifelse_centered_only!(out, grid, scheme, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_ifelse_centered_only!, out, grid, scheme, c)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Two-level nested ifelse with Centered-only
#   ifelse(halo_outer, centered, ifelse(halo_inner, centered, centered))
#   Tests the nesting depth with simple math.
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_nested_ifelse_centered!(out, grid, scheme_outer, scheme_inner, c)
    i, j, k = @index(Global, NTuple)
    centered = Centered(order=2)
    @inbounds out[i, j, k] = ifelse(
        outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme_outer),
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, centered, LeftBias(), c),
        ifelse(
            outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme_inner),
            biased_interpolate_xᶠᵃᵃ(i, j, k, grid, centered, LeftBias(), c),
            biased_interpolate_xᶠᵃᵃ(i, j, k, grid, centered, LeftBias(), c)))
end

function run_nested_ifelse_centered!(out, grid, scheme_outer, scheme_inner, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_nested_ifelse_centered!, out, grid, scheme_outer, scheme_inner, c)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Run all probes
# ═══════════════════════════════════════════════════════════════════════════

grid, c, out = build_bpb()
scheme = WENO(order=5)  # WENO{3, Float64, Float32}

t1 = probe("1: WENO{3} direct (no ifelse)",
           run_weno3_direct!, out, grid, scheme, c)

t2 = probe("2: full _biased_interpolate_xᶠᵃᵃ (topology dispatch)",
           run_full_topo_dispatch!, out, grid, scheme, c)

t3 = probe("3: single ifelse(halo, WENO{3}, Centered{1})",
           run_single_ifelse!, out, grid, scheme, Centered(order=2), c)

t4 = probe("4: single ifelse with Centered only (no WENO math)",
           run_ifelse_centered_only!, out, grid, scheme, c)

t5 = probe("5: nested ifelse with Centered only",
           run_nested_ifelse_centered!, out, grid, scheme, scheme.buffer_scheme, c)

@info "═══ Systematic isolation results ═══"
for (name, result) in [
    ("1: WENO{3} direct (no ifelse)           ", t1),
    ("2: full topology dispatch                ", t2),
    ("3: single ifelse(WENO{3}, Centered)      ", t3),
    ("4: single ifelse Centered-only           ", t4),
    ("5: nested ifelse Centered-only           ", t5),
]
    status = result.ok ? "PASS" : "FAIL"
    @info "$name : $status"
    if !result.ok
        errmsg = first(result.error, 300)
        @info "  Error (truncated): $errmsg"
    end
end

@info """

Interpretation guide:
  t1 PASS, t2 FAIL → the topology ifelse wrapping is needed to trigger the bug
  t3 FAIL          → single-level ifelse + WENO is sufficient
  t3 PASS          → need nested buffer recursion (two levels)
  t4 FAIL          → ifelse + Oceananigans field access alone triggers it (not WENO)
  t4 PASS          → WENO math contributes to the failure
  t5 FAIL          → nested ifelse + Oceananigans fields, even without WENO
  t5 PASS          → the nesting + WENO math together are needed
"""
