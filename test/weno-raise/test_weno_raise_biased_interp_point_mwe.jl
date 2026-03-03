using Test
using Random

using Oceananigans
using Reactant

using Oceananigans.Advection: WENO, LeftBias, _biased_interpolate_xᶠᵃᵃ
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField
using Oceananigans.Grids: Bounded, Periodic


"""
Ultra-minimal probe for the WENO bounded-x raise failure:
directly compile a scalar wrapper around `_biased_interpolate_xᶠᵃᵃ`.
"""
@inline function biased_interp_point(i, j, k, grid, scheme, c)
    return _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c)
end

function run_point_probe(; topology, order)
    arch = ReactantState()
    grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1), topology)
    c = CenterField(grid)

    Random.seed!(20260303)
    set!(c, randn(size(c)...))

    # i=1 is the x-boundary-sensitive point for BPB topology.
    i, j, k = 1, 1, 1
    scheme = WENO(order=order)

    @info "Compiling scalar WENO interpolation probe" topology order i j k
    compiled_point = @compile raise=true raise_first=true sync=true biased_interp_point(i, j, k, grid, scheme, c)

    @info "Running scalar WENO interpolation probe" topology order
    return compiled_point(i, j, k, grid, scheme, c)
end

@testset "WENO raise scalar point probe" begin
    # Core signal: bounded x fails.
    @test run_point_probe(; topology=(Bounded, Periodic, Bounded), order=5) isa Number

    # Control: fully periodic should avoid the bounded-x trigger.
    @test run_point_probe(; topology=(Periodic, Periodic, Periodic), order=5) isa Number
end
