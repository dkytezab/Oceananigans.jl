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

# ── Baseline: current outside_biased_halo_xᶠ (KNOWN FAIL) ──

@kernel function kernel_baseline!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = outside_biased_halo_xᶠ(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_baseline!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_baseline!, out, grid, scheme, c)
    return nothing
end

# ── W1: pre-compute bounds outside the kernel ──

@kernel function kernel_w1!(out, lo, hi, c)
    i, j, k = @index(Global, NTuple)
    cond = (i >= lo) & (i <= hi)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w1!(out, lo, hi, c, grid)
    launch!(architecture(grid), grid, :xyz, kernel_w1!, out, lo, hi, c)
    return nothing
end

# ── W2: simplified 2-condition predicate (inline, same logic) ──

@inline simplified_halo_x(i, N, H) = (i >= H + 1) & (i <= N + 1 - H)

@kernel function kernel_w2!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    H = required_halo_size_x(scheme)
    cond = simplified_halo_x(i, grid.Nx, H)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w2!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_w2!, out, grid, scheme, c)
    return nothing
end

# ── W3: pass grid.Nx as a separate kernel argument (not through the struct) ──

@kernel function kernel_w3!(out, Nx, scheme, c)
    i, j, k = @index(Global, NTuple)
    H = required_halo_size_x(scheme)
    cond = (i >= H + 1) & (i <= Nx + 1 - (H - 1)) & (i >= H) & (i <= Nx + 1 - H)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w3!(out, Nx, scheme, c, grid)
    launch!(architecture(grid), grid, :xyz, kernel_w3!, out, Nx, scheme, c)
    return nothing
end

# ── W4: same as outside_biased_halo_xᶠ body but directly inlined ──

@kernel function kernel_w4!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    H = required_halo_size_x(scheme)
    cond = (i >= H + 1) & (i <= N + 1 - (H - 1)) & (i >= H) & (i <= N + 1 - H)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w4!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_w4!, out, grid, scheme, c)
    return nothing
end

# ── W5: same logic but with topology dispatch, without calling outside_biased_halo_xᶠ ──

@inline my_halo_check(i, ::Type{Bounded}, N, adv) =
    (i >= required_halo_size_x(adv) + 1) & (i <= N + 1 - (required_halo_size_x(adv) - 1)) &
    (i >= required_halo_size_x(adv))     & (i <= N + 1 - required_halo_size_x(adv))

@kernel function kernel_w5!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    cond = my_halo_check(i, topology(grid, 1), grid.Nx, scheme)
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w5!(out, grid, scheme, c)
    launch!(architecture(grid), grid, :xyz, kernel_w5!, out, grid, scheme, c)
    return nothing
end

# ── W6: use Nx from grid but pass through Val to force constant ──

@inline halo_check_val(i, ::Type{Bounded}, ::Val{N}, ::Val{H}) where {N, H} =
    (i >= H + 1) & (i <= N + 1 - (H - 1)) & (i >= H) & (i <= N + 1 - H)

@kernel function kernel_w6!(out, grid, ::Val{H}, c) where H
    i, j, k = @index(Global, NTuple)
    cond = halo_check_val(i, topology(grid, 1), Val(grid.Nx), Val(H))
    @inbounds out[i, j, k] = ifelse(cond, c[i, j, k], c[i, j, k] + 1.0)
end

function run_w6!(out, grid, valH, c)
    launch!(architecture(grid), grid, :xyz, kernel_w6!, out, grid, valH, c)
    return nothing
end

function probe(label, run_fn!, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        @info "  → PASS" label
        return (label=label, ok=true)
    catch err
        buf = IOBuffer(); showerror(buf, err)
        @info "  → FAIL" label
        return (label=label, ok=false, error=first(String(take!(buf)), 200))
    end
end

H = required_halo_size_x(scheme)

probe("baseline: outside_biased_halo_xᶠ",  run_baseline!, out, grid, scheme, c)
probe("W1: pre-computed lo/hi as args",     run_w1!, out, H+1, grid.Nx+1-H, c, grid)
probe("W2: simplified_halo_x(i, N, H)",     run_w2!, out, grid, scheme, c)
probe("W3: Nx passed separately (not struct)", run_w3!, out, grid.Nx, scheme, c, grid)
probe("W4: inline body, N=grid.Nx in kernel", run_w4!, out, grid, scheme, c)
probe("W5: my_halo_check (identical body, fresh function)", run_w5!, out, grid, scheme, c)
probe("W6: Val{N}/Val{H} forced constants", run_w6!, out, grid, Val(H), c)
