using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index, StaticSize

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)

# ─── Part 1: Oceananigans version (EXPECTED FAIL) ───

@kernel function ocean_kernel!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end

function ocean_run!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, ocean_kernel!, out, grid, c)
    return nothing
end

@info "Part 1: Oceananigans (grid.Nx in kernel, launch! with StaticSize)"
try
    f! = @compile raise=true raise_first=true sync=true ocean_run!(out, grid, c)
    f!(out, grid, c)
    @info "  → PASS"
catch
    @info "  → FAIL"
end

# ─── Part 2: Pure Reactant — no Oceananigans ───

const RKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const RBackend = RKAExt.ReactantBackend

struct Params
    Nx::Int
end

@kernel function pure_kernel!(out, p, c)
    i, j, k = @index(Global, NTuple)
    N = p.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end

function pure_run!(out, p, c)
    pure_kernel!(RBackend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, p, c)
    return nothing
end

p      = Params(4)
c_ra   = Reactant.ConcreteRArray(rand(Float64, 4, 4, 4))
out_ra = Reactant.ConcreteRArray(zeros(Float64, 4, 4, 4))

@info "Part 2: Pure Reactant (struct.Nx, ReactantBackend, StaticSize((16,16)))"
try
    f! = @compile raise=true raise_first=true sync=true pure_run!(out_ra, p, c_ra)
    f!(out_ra, p, c_ra)
    @info "  → PASS"
catch
    @info "  → FAIL"
end
