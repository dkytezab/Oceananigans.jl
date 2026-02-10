# MWE: compute_source_term! fails with raise=true due to Float64 → ComplexF64 store
# Reproduces B.6.7 Issue 1 using the actual Oceananigans kernel and solver.

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Models.NonhydrostaticModels: compute_source_term!
using Reactant
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1),
                       halo=(3, 3), topology=(Periodic, Periodic, Flat))

model  = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
solver = model.pressure_solver
Ũ      = model.velocities
Δt     = 1.0

function go!(solver, Ũ)
    compute_source_term!(solver, nothing, Ũ, Δt)
    return nothing
end

Reactant.@compile sync=true go!(solver, Ũ)
