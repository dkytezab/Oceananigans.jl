include("../reactant_test_utils.jl")
using Reactant: @trace

using Random

@testset "WENO raise BPB MWE (normal raise style)" begin
    arch = ReactantState()
    @info "Starting WENO BPB MWE" architecture=arch

    # BPB = (Bounded, Periodic, Bounded)
    @info "Constructing BPB grid"
    grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1),
                           topology=(Bounded, Periodic, Bounded))

    @info "Constructing HydrostaticFreeSurfaceModel with WENO tracer advection"
    model = HydrostaticFreeSurfaceModel(grid;
                free_surface = ExplicitFreeSurface(),
                timestepper  = :QuasiAdamsBashforth2,
                buoyancy     = nothing,
                tracers      = :c,
                tracer_advection = WENO())

    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for _ in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    # Ensure tracer advection is exercised with nontrivial flow and tracer state.
    @info "Initializing velocity and tracer fields"
    Random.seed!(20260303)
    u_init = randn(size(model.velocities.u)...)
    v_init = randn(size(model.velocities.v)...)
    c_init = randn(size(model.tracers.c)...)
    set!(model, u=u_init, v=v_init, c=c_init)

    @testset "Compiled time_step! (raise=true, currently broken for WENO BPB)" begin
        Δt = 0.001
        Nt = 4
        @info "Attempting raise=true compilation" Δt Nt
        @test_broken begin
            compiled_run_raise! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            @info "Running raise=true compiled function"
            compiled_run_raise!(model, Δt, Nt)
            model.clock.iteration == Nt
        end
    end

    @testset "Compiled time_step! (no-raise workaround)" begin
        Δt = 0.001
        Nt = 4
        @info "Compiling no-raise workaround" Δt Nt
        compiled_run! = @compile sync=true run_timesteps!(model, Δt, Nt)
        @info "Running no-raise compiled function"
        compiled_run!(model, Δt, Nt)
        @test model.clock.iteration == Nt
    end
end
