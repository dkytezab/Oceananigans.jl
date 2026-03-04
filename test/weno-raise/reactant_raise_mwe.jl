using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

@kernel function my_kernel!(out, c, Nx)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i <= Nx ÷ 2, c[i, j, k], c[i, j, k] + 1.0)
end

function run!(out, c, Nx, grid)
    launch!(architecture(grid), grid, :xyz, my_kernel!, out, c, Nx)
    return nothing
end

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), halo=(3, 3, 3),
                       extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
c   = CenterField(grid)
out = XFaceField(grid)
set!(c, (x, y, z) -> x + y)

compiled! = @compile raise=true raise_first=true sync=true run!(out, c, 4, grid)
compiled!(out, c, 4, grid)
