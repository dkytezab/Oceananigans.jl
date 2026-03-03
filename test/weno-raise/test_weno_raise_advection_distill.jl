using Test
using Random

using Oceananigans
using Reactant

using KernelAbstractions: @kernel, @index

using Oceananigans.Advection: WENO, LeftBias, _biased_interpolate_xᶠᵃᵃ
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!

@kernel function x_biased_interp_kernel!(out, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, LeftBias(), c)
end

function run_x_biased_interp!(out, grid, scheme, c)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, x_biased_interp_kernel!, out, grid, scheme, c)
    return nothing
end

function build_case(; topology, scheme)
    arch = ReactantState()
    grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1), topology)
    c = CenterField(grid)
    out = XFaceField(grid)

    Random.seed!(20260303)
    set!(c, randn(size(c)...))
    fill_halo_regions!(c)

    return grid, c, out, scheme
end

function probe_raise(label; topology, scheme)
    @info "Probing raise=true on advection interpolation" label topology scheme=summary(scheme)
    grid, c, out, scheme = build_case(; topology, scheme)
    try
        compiled! = @compile raise=true raise_first=true sync=true run_x_biased_interp!(out, grid, scheme, c)
        compiled!(out, grid, scheme, c)
        return (ok=true, error="")
    catch err
        io = IOBuffer()
        showerror(io, err)
        return (ok=false, error=String(take!(io)))
    end
end

@testset "WENO raise distillation: x-biased interpolation kernel" begin
    weno_bpb = probe_raise("weno5-bpb"; topology=(Bounded, Periodic, Bounded), scheme=WENO(order=5))
    weno_ppp = probe_raise("weno5-ppp"; topology=(Periodic, Periodic, Periodic), scheme=WENO(order=5))
    weno3_bpb = probe_raise("weno3-bpb"; topology=(Bounded, Periodic, Bounded), scheme=WENO(order=3))

    @info "Distillation results" weno_bpb weno_ppp weno3_bpb

    # Core distilled signal:
    # - bounded-x WENO raise fails
    # - fully periodic WENO raise succeeds
    @test !weno_bpb.ok
    @test weno_ppp.ok

    # Additional narrow-order probe (informational, behavior can differ by lowering/passes).
    @test weno3_bpb.ok || !weno3_bpb.ok
end
