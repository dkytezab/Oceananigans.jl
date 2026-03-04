using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)

# @kernel function kernel!(out, grid, c)
#     i, j, k = @index(Global, NTuple)
#     N = grid.Nx
#     @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
#                                     c[i, j, k], c[i, j, k] + 1.0)
# end

@kernel function kernel!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 1) & (i <= N - 2),
                                    c[i, j, k], c[i, j, k] + 1.0)
end

function run!(out, grid, c)
    launch!(architecture(grid), grid, :xyz, kernel!, out, grid, c)
    return nothing
end

f! = @compile raise=true raise_first=true sync=true run!(out, grid, c)
f!(out, grid, c)
