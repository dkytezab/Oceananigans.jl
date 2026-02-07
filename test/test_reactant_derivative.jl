#####
##### Integration tests for Reactant/Enzyme automatic differentiation
##### with NonhydrostaticModel (2D Periodic)
#####
# This test follows the canonical pattern from differentiability-mwe.mdc
# Testing gradient computation through time-stepping with FFT-based pressure solver

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Test
using CUDA  # Required for ReactantCUDA extension

Reactant.set_default_backend("cpu")

@testset "Reactant/Enzyme differentiation - NonhydrostaticModel 2D Periodic" begin
    @info "Testing Reactant/Enzyme differentiation with NonhydrostaticModel..."
    
    #####
    ##### Grid and Model Setup
    #####
    
    @time "Constructing grid" grid = RectilinearGrid(ReactantState();
        size = (8, 8),
        extent = (1.0, 1.0),
        halo = (3, 3),
        topology = (Periodic, Periodic, Flat)
    )
    
    @time "Constructing model" model = NonhydrostaticModel(grid; 
        timestepper = :QuasiAdamsBashforth2  # QB2 is stable; use RK3 if testing that
    )
    
    @time "Creating shadow model (make_zero)" dmodel = Enzyme.make_zero(model)
    
    #####
    ##### Initial condition fields
    #####
    
    @time "Constructing initial condition field" begin
        u_init = XFaceField(grid)
        # Simple linear initial condition for u velocity
        set!(u_init, (x, y) -> 0.01 * sin(2π * x) * cos(2π * y))
    end
    
    @time "Constructing shadow field" begin
        du_init = XFaceField(grid)
        set!(du_init, 0.0)
    end
    
    #####
    ##### Loss function: mean squared velocity after time-stepping
    #####
    
    function loss(model, u_init, Δt, nsteps)
        # Set initial condition
        # NOTE: enforce_incompressibility=false avoids FFT solver call during set!
        # which causes complex<f64> MLIR issues. Pressure correction still happens
        # during time_step! so physics is unchanged.
        set!(model, u=u_init; enforce_incompressibility=false)
        
        # Time-step with @trace (required for AD through loops)
        # track_numbers=false is REQUIRED for clock compatibility
        # nsteps must be a perfect square when checkpointing=true
        @trace track_numbers=false mincut=true checkpointing=true for i in 1:nsteps
            time_step!(model, Δt)
        end
        
        # Loss: mean squared u velocity
        return mean(interior(model.velocities.u).^2)
    end
    
    #####
    ##### Gradient function using Enzyme reverse-mode AD
    #####
    
    function grad_loss(model, dmodel, u_init, du_init, Δt, nsteps)
        # Reset shadow field to zero
        parent(du_init) .= 0
        
        # Compute gradient using reverse-mode AD
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss,
            Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(u_init, du_init),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps)
        )
        
        return du_init, loss_value
    end
    
    #####
    ##### Test parameters
    #####
    
    Δt = 0.001
    nsteps = 4  # Must be a perfect square for checkpointing=true (4 = 2²)
    
    #####
    ##### Tests
    #####
    
    @testset "Forward pass compilation" begin
        @info "  Testing forward pass compilation..."
        @time "Compiling loss" compiled_loss = Reactant.@compile raise_first=true raise=true sync=true loss(
            model, u_init, Δt, nsteps)
        @test compiled_loss !== nothing
        
        @time "Running compiled loss" result = compiled_loss(model, u_init, Δt, nsteps)
        @test !isnan(result)
        @test result >= 0
        @info "    Forward loss value: $result"
    end
    
    @testset "Gradient compilation" begin
        @info "  Testing gradient compilation..."
        @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, u_init, du_init, Δt, nsteps)
        @test compiled_grad !== nothing
    end
    
    @testset "Gradient computation" begin
        @info "  Testing gradient computation..."
        
        # Reset model state
        # NOTE: enforce_incompressibility=false is REQUIRED when calling set! outside
        # of Reactant compilation, because the FFT-based pressure solver cannot run
        # eagerly on ConcretePJRTArray (would trigger ComplexF64 conversion error)
        set!(model, u=0.0, v=0.0, w=0.0; enforce_incompressibility=false)
        model.clock.time = 0.0
        model.clock.iteration = 0
        
        # Compile and run gradient function
        @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, u_init, du_init, Δt, nsteps)
        
        @time "Running compiled gradient" du, loss_val = compiled_grad(
            model, dmodel, u_init, du_init, Δt, nsteps)
        
        @info "    Loss value: $loss_val"
        @info "    Max |gradient|: $(maximum(abs, interior(du)))"
        
        # Verify loss is valid
        @test loss_val >= 0
        @test !isnan(loss_val)
        
        # Verify gradient is non-zero (if loss > 0, gradient should be non-zero)
        grad_max = maximum(abs, interior(du))
        @info "    Gradient max magnitude: $grad_max"
        
        if loss_val > 0
            @test grad_max > 0  # Gradient should be non-zero for non-zero loss
        end
        
        # Verify no NaN in gradient
        @test !any(isnan, interior(du))
    end
end
