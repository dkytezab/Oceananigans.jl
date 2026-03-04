using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture, topology
using Oceananigans.Advection: outside_biased_halo_xᶠ, WENO
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)
scheme = WENO(order=5)

# ── Test A: simple ifelse(i <= 2, ...) — known to PASS ──

@kernel function kernel_a!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i <= 2, c[i, j, k], c[i, j, k] + 1.0)
end

function run_a!(out, c, grid)
    launch!(architecture(grid), grid, :xyz, kernel_a!, out, c)
    return nothing
end

# ── Test B: manual 4-way AND like outside_biased_halo_xᶠ with constants ──

@kernel function kernel_b!(out, c)
    i, j, k = @index(Global, NTuple)
    cond = (i >= 4) & (i <= 3) & (i >= 3) & (i <= 2)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_b!(out, c, grid)
    launch!(architecture(grid), grid, :xyz, kernel_b!, out, c)
    return nothing
end

# ── Test C: actual outside_biased_halo_xᶠ call ──

@kernel function kernel_c!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_c!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_c!, out, grid, scheme, c)
    return nothing
end

# ── Test D: just ifelse with WENO interpolation (no halo predicate) ──
# This is t1 from systematic — should pass as control.

using Oceananigans.Advection: biased_interpolate_xᶠᵃᵃ, LeftBias

@kernel function kernel_d!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c)
end

function run_d!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_d!, out, grid, scheme, c)
    return nothing
end

# ── Test E: halo predicate + WENO interpolation (the real failing combo) ──

@kernel function kernel_e!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond,
        biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c),
        c[i, j, k] + 1.0)
end

function run_e!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_e!, out, grid, scheme, c)
    return nothing
end

function probe(label, run_fn!, args...)
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        println("$label : PASS")
        return true
    catch err
        buf = IOBuffer(); showerror(buf, err)
        println("$label : FAIL")
        println("  ", first(String(take!(buf)), 200))
        return false
    end
end

probe("A: ifelse(i<=2, load, load+1)",              run_a!, out, c, grid)
probe("B: ifelse(4-way AND constants, load, load+1)", run_b!, out, c, grid)
probe("C: ifelse(outside_biased_halo, load, load+1)", run_c!, out, grid, scheme, c)
probe("D: WENO direct (no ifelse, no halo check)",   run_d!, out, grid, scheme, c)
probe("E: ifelse(outside_biased_halo, WENO, load+1)", run_e!, out, grid, scheme, c)
