#=
MedWE: Oceananigans pressure solve kernels with ComplexF64 storage

Launches the actual _compute_source_term! and copy_real_component! kernels
from Oceananigans using launch! with ComplexF64 storage, exactly as
solve_for_pressure! and solve! do in the FFT pressure pipeline.
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Utils: launch!
using Oceananigans.Fields: interior, offset_compute_index
using Oceananigans.Solvers: copy_real_component!
using Reactant
using Test
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size = (4, 4),
    extent = (1.0, 1.0),
    halo = (3, 3),
    topology = (Periodic, Periodic, Flat)
)

# Build a NonhydrostaticModel to get the actual solver and fields
model = NonhydrostaticModel(grid; timestepper = :QuasiAdamsBashforth2)

solver  = model.pressure_solver
rhs     = solver.storage          # ComplexF64 array
pressure = model.pressures.pNHS   # Float64 field
Ũ       = model.velocities        # velocity fields

#####
##### Issue 1: _compute_source_term! — the actual Oceananigans kernel
##### (solve_for_pressure.jl:108-113)
#####

using Oceananigans.Models.NonhydrostaticModels: compute_source_term!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!

function test_compute_source_term!(solver, Ũ)
    compute_source_term!(solver, nothing, Ũ, 1.0)
    return nothing
end

@testset "Issue 1: compute_source_term! (Float64 → ComplexF64)" begin
    @test_broken begin
        compiled = Reactant.@compile raise=true sync=true test_compute_source_term!(solver, Ũ)
        compiled(solver, Ũ)
        true
    end
end

#####
##### Issue 2: copy_real_component! — the actual Oceananigans kernel
##### (fft_based_poisson_solver.jl:121, 128-136)
#####

function test_copy_real_component!(pressure, solver, grid)
    ϕc = solver.storage
    launch!(grid.architecture, grid, :xyz, copy_real_component!, pressure, ϕc, indices(pressure))
    return nothing
end

@testset "Issue 2: copy_real_component! (real on ComplexF64)" begin
    @test_broken begin
        compiled = Reactant.@compile raise=true sync=true test_copy_real_component!(pressure, solver, grid)
        compiled(pressure, solver, grid)
        true
    end
end

#####
##### Broadcast workarounds
#####

function broadcast_source_term!(solver, Ũ)
    # Instead of KA kernel, use broadcast to write into ComplexF64 storage
    solver.storage .= Complex.(0.0, 0.0)
    return nothing
end

function broadcast_copy_real!(pressure, solver)
    parent(pressure) .= real.(solver.storage)
    return nothing
end

@testset "Broadcast workaround: source term" begin
    compiled = Reactant.@compile raise=true sync=true broadcast_source_term!(solver, Ũ)
    compiled(solver, Ũ)
    @test true
end

@testset "Broadcast workaround: copy real component" begin
    compiled = Reactant.@compile raise=true sync=true broadcast_copy_real!(pressure, solver)
    compiled(pressure, solver)
    @test true
end
