using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index, StaticSize

const RKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const RBackend = RKAExt.ReactantBackend

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)

struct Params
    Nx::Int
end
p      = Params(4)
c_ra   = Reactant.ConcreteRArray(rand(Float64, 4, 4, 4))
out_ra = Reactant.ConcreteRArray(zeros(Float64, 4, 4, 4))

function probe(label, run_fn!, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        @info "  → PASS" label
    catch
        @info "  → FAIL" label
    end
end

# ─── Oceananigans: 2 conditions ───

@kernel function ok2!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end
function rok2!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, ok2!, out, grid, c); nothing
end

# ─── Oceananigans: 3 conditions ───

@kernel function ok3!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end
function rok3!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, ok3!, out, grid, c); nothing
end

# ─── Pure Reactant: 2 conditions ───

@kernel function pk2!(out, p, c)
    i, j, k = @index(Global, NTuple)
    N = p.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end
function rpk2!(out, p, c)
    pk2!(RBackend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, p, c); nothing
end

# ─── Pure Reactant: 3 conditions ───

@kernel function pk3!(out, p, c)
    i, j, k = @index(Global, NTuple)
    N = p.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end
function rpk3!(out, p, c)
    pk3!(RBackend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, p, c); nothing
end

probe("Ocean 2-cond: (i>=4) & (i<=N-2)",          rok2!, out, grid, c)
probe("Ocean 3-cond: (i>=4) & (i<=N-1) & (i<=N-2)", rok3!, out, grid, c)
probe("Pure  2-cond: (i>=4) & (i<=N-2)",          rpk2!, out_ra, p, c_ra)
probe("Pure  3-cond: (i>=4) & (i<=N-1) & (i<=N-2)", rpk3!, out_ra, p, c_ra)
