using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture, topology
using Oceananigans.Advection: outside_biased_halo_xᶠ, WENO, required_halo_size_x
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)
scheme = WENO(order=5)

function probe(label, run_fn!, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        @info "  → PASS" label
        return true
    catch err
        @info "  → FAIL" label
        return false
    end
end

# ── B: baseline passing test — literal constants, no grid/scheme args ──

@kernel function kernel_b!(out, c)
    i, j, k = @index(Global, NTuple)
    cond = (i >= 4) & (i <= 3) & (i >= 3) & (i <= 2)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_b!(out, c, grid)
    launch!(architecture(grid), grid, :xyz, kernel_b!, out, c)
    return nothing
end

# ── S1: pass grid to kernel but still use literal constants ──

@kernel function kernel_s1!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    cond = (i >= 4) & (i <= 3) & (i >= 3) & (i <= 2)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_s1!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, kernel_s1!, out, grid, c)
    return nothing
end

# ── S2: compute constants from grid.Nx inside the kernel ──

@kernel function kernel_s2!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    cond = (i >= 4) & (i <= N - 1) & (i >= 3) & (i <= N - 2)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_s2!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, kernel_s2!, out, grid, c)
    return nothing
end

# ── S3: compute all constants from grid.Nx and H=required_halo_size_x ──

@kernel function kernel_s3!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    H = required_halo_size_x(scheme)
    cond = (i >= H + 1) & (i <= N + 1 - (H - 1)) & (i >= H) & (i <= N + 1 - H)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_s3!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_s3!, out, grid, scheme, c)
    return nothing
end

# ── S4: same as S3 but factored into a local @inline function ──

@inline my_halo_pred(i, N, H) =
    (i >= H + 1) & (i <= N + 1 - (H - 1)) & (i >= H) & (i <= N + 1 - H)

@kernel function kernel_s4!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = my_halo_pred(i, grid.Nx, required_halo_size_x(scheme))
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_s4!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_s4!, out, grid, scheme, c)
    return nothing
end

# ── S5: same as S4 but with Bounded type dispatch (like outside_biased_halo_xᶠ) ──

@inline my_halo_dispatch(i, ::Type{Bounded}, N, adv) =
    (i >= required_halo_size_x(adv) + 1) & (i <= N + 1 - (required_halo_size_x(adv) - 1)) &
    (i >= required_halo_size_x(adv))     & (i <= N + 1 - required_halo_size_x(adv))

@kernel function kernel_s5!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = my_halo_dispatch(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_s5!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_s5!, out, grid, scheme, c)
    return nothing
end

# ── C: actual outside_biased_halo_xᶠ (KNOWN FAIL) ──

@kernel function kernel_c!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end
function run_c!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_c!, out, grid, scheme, c)
    return nothing
end

probe("B:  literal (i>=4)&(i<=3)&(i>=3)&(i<=2)",         run_b!,  out, c, grid)
probe("S1: B + grid as kernel arg (unused)",              run_s1!, out, grid, c)
probe("S2: constants from grid.Nx inside kernel",         run_s2!, out, grid, c)
probe("S3: constants from grid.Nx + required_halo_size",  run_s3!, out, grid, scheme, c)
probe("S4: S3 factored into @inline function",            run_s4!, out, grid, scheme, c)
probe("S5: S4 + Bounded type dispatch",                   run_s5!, out, grid, scheme, c)
probe("C:  actual outside_biased_halo_xᶠ",               run_c!,  out, grid, scheme, c)
