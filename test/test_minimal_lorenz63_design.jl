using Test
using Enzyme
using Reactant
using CairoMakie
using Random

"""
Minimal differentiable sensor-design toy:
- Dynamics: Lorenz-63 with forward Euler.
- Design variable `d`: 3-vector observation scaling.
- Objective: posterior sample variance surrogate of scalar QoI.
"""

@inline function lorenz63_euler_step(x1, x2, x3, dt)
    σ = 10.0
    ρ = 28.0
    β = 8.0 / 3.0
    dx1 = σ * (x2 - x1)
    dx2 = x1 * (ρ - x3) - x2
    dx3 = x1 * x2 - β * x3
    return x1 + dt * dx1, x2 + dt * dx2, x3 + dt * dx3
end

function lorenz63_rollout(x0::NTuple{3, Float64}, dt, nsteps)
    x1, x2, x3 = x0
    @inbounds for _ in 1:nsteps
        x1, x2, x3 = lorenz63_euler_step(x1, x2, x3, dt)
    end
    return x1, x2, x3
end

function sample_initial_ensemble(M::Int; μ = (1.0, 1.0, 1.0), σ = 0.15, rng = Random.default_rng())
    return NTuple{3, Float64}[
        (μ[1] + σ * randn(rng),
         μ[2] + σ * randn(rng),
         μ[3] + σ * randn(rng)) for _ in 1:M
    ]
end

@inline observation_weighted_components(x1, x2, x3, d::NTuple{3, Float64}) = d[1] * x1 + d[2] * x2 + d[3] * x3

function minimal_design_objective(d::NTuple{3, Float64}, ensemble::Vector{NTuple{3, Float64}}, dt, nsteps)
    M = length(ensemble)

    # Pass 1: means of forecast QoI and observed quantity.
    z̄ = 0.0
    h̄ = 0.0
    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        z̄ += x1
        h̄ += observation_weighted_components(x1, x2, x3, d)
    end
    z̄ /= M
    h̄ /= M

    # Pass 2: regression gain K = cov(z,h) / (var(h) + R(d)).
    cov_zh = 0.0
    var_h = 0.0
    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        zδ = x1 - z̄
        hδ = observation_weighted_components(x1, x2, x3, d) - h̄
        cov_zh += zδ * hδ
        var_h += hδ * hδ
    end
    cov_zh /= (M - 1)
    var_h /= (M - 1)

    R = 0.05 + d[1]^2 + d[2]^2 + d[3]^2
    K = cov_zh / (var_h + R)

    # Pass 3: posterior variance surrogate.
    z̄a = 0.0
    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        za = x1 - K * (observation_weighted_components(x1, x2, x3, d) - h̄)
        z̄a += za
    end
    z̄a /= M

    var_za = 0.0
    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        za = x1 - K * (observation_weighted_components(x1, x2, x3, d) - h̄)
        δ = za - z̄a
        var_za += δ * δ
    end

    return var_za / (M - 1)
end

function design_diagnostics(d::NTuple{3, Float64}, ensemble::Vector{NTuple{3, Float64}}, dt, nsteps)
    M = length(ensemble)
    z̄ = 0.0
    h̄ = 0.0

    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        z̄ += x1
        h̄ += observation_weighted_components(x1, x2, x3, d)
    end
    z̄ /= M
    h̄ /= M

    cov_zh = 0.0
    var_h = 0.0
    @inbounds for x0 in ensemble
        x1, x2, x3 = lorenz63_rollout(x0, dt, nsteps)
        zδ = x1 - z̄
        hδ = observation_weighted_components(x1, x2, x3, d) - h̄
        cov_zh += zδ * hδ
        var_h += hδ * hδ
    end
    cov_zh /= (M - 1)
    var_h /= (M - 1)

    R = 0.05 + d[1]^2 + d[2]^2 + d[3]^2
    K = cov_zh / (var_h + R)
    J = minimal_design_objective(d, ensemble, dt, nsteps)

    return (; M, z̄, h̄, cov_zh, var_h, R, K, J)
end

"""
Return loss and ∂loss/∂d using Enzyme reverse mode.

`Duplicated` is not needed here because we only differentiate with respect to scalar
design components (`Active(dᵢ)`), while ensemble and integration settings are constants.
"""
@inline minimal_design_objective_components(d1, d2, d3, ensemble, dt, nsteps) =
    minimal_design_objective((d1, d2, d3), ensemble, dt, nsteps)

function first_real_gradient(x)
    if x isa Real
        return Float64(x)
    elseif x isa Tuple
        @inbounds for xi in x
            gi = first_real_gradient(xi)
            gi === nothing || return gi
        end
    end
    return nothing
end

function loss_and_gradient(d::NTuple{3, Float64}, ensemble::Vector{NTuple{3, Float64}}, dt, nsteps)
    d1, d2, d3 = d

    grad1_pack, loss = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        minimal_design_objective_components,
        Active(d1),
        Const(d2),
        Const(d3),
        Const(ensemble),
        Const(dt),
        Const(nsteps),
    )

    grad2_pack, _ = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        minimal_design_objective_components,
        Const(d1),
        Active(d2),
        Const(d3),
        Const(ensemble),
        Const(dt),
        Const(nsteps),
    )

    grad3_pack, _ = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        minimal_design_objective_components,
        Const(d1),
        Const(d2),
        Active(d3),
        Const(ensemble),
        Const(dt),
        Const(nsteps),
    )

    g1 = first_real_gradient(grad1_pack)
    g2 = first_real_gradient(grad2_pack)
    g3 = first_real_gradient(grad3_pack)
    @assert !isnothing(g1) && !isnothing(g2) && !isnothing(g3)
    return loss, (g1, g2, g3)
end

function optimize_design_gradient_descent(
    d0::NTuple{3, Float64},
    ensemble::Vector{NTuple{3, Float64}},
    dt,
    nsteps;
    η = 0.5,
    niter = 25,
)
    d = collect(d0)
    d_history = NTuple{3, Float64}[Tuple(d)]
    loss_history = Float64[]
    grad_history = NTuple{3, Float64}[]

    for _ in 1:niter
        loss, grad = loss_and_gradient(Tuple(d), ensemble, dt, nsteps)
        push!(loss_history, loss)
        push!(grad_history, grad)
        d[1] -= η * grad[1]
        d[2] -= η * grad[2]
        d[3] -= η * grad[3]
        push!(d_history, Tuple(d))
    end

    return (; d_final = Tuple(d), d_history, loss_history, grad_history)
end

function optimize_design_adam(
    d0::NTuple{3, Float64},
    ensemble::Vector{NTuple{3, Float64}},
    dt,
    nsteps;
    η = 5e-2,
    β1 = 0.9,
    β2 = 0.999,
    ϵ = 1e-8,
    niter = 100,
)
    d = collect(d0)
    m = zeros(3)
    v = zeros(3)
    d_history = NTuple{3, Float64}[Tuple(d)]
    loss_history = Float64[]
    grad_history = NTuple{3, Float64}[]

    for t in 1:niter
        loss, grad = loss_and_gradient(Tuple(d), ensemble, dt, nsteps)
        g = [grad[1], grad[2], grad[3]]
        push!(loss_history, loss)
        push!(grad_history, grad)

        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* (g .^ 2)
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)

        d .-= η .* m̂ ./ (sqrt.(v̂) .+ ϵ)
        push!(d_history, Tuple(d))
    end

    return (; d_final = Tuple(d), d_history, loss_history, grad_history)
end

function plot_loss_history(loss_history::AbstractVector{<:Real}; optimizer_name = "Optimizer")
    fig = Figure(size = (700, 380))
    ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel = "Loss (posterior QoI variance)",
              title = "Lorenz-63 Sensor-Design Loss During " * optimizer_name)
    lines!(ax, 1:length(loss_history), loss_history, linewidth = 3)
    scatter!(ax, 1:length(loss_history), loss_history, markersize = 8)
    return fig
end

@testset "Minimal Lorenz-63 differentiable design (Reactant + Enzyme)" begin
    rng = MersenneTwister(1234)
    M = 1000
    ensemble = sample_initial_ensemble(M; μ = (5.0, 5.0, 5.0), σ = 10, rng)

    dt = 0.01
    nsteps = 1500
    d0 = (0.7, 0.2, -0.1)

    J = minimal_design_objective(d0, ensemble, dt, nsteps)
    @test isfinite(J)
    @test J > 0

    # Reactant: ensure this tiny objective can be jitted.
    J_reactant = @jit minimal_design_objective(d0, ensemble, dt, nsteps)
    @test J_reactant ≈ J rtol = 1e-10

    # Enzyme: gradient should match finite differences.
    ϵ = 1e-6
    Jp1 = minimal_design_objective((d0[1] + ϵ, d0[2], d0[3]), ensemble, dt, nsteps)
    Jm1 = minimal_design_objective((d0[1] - ϵ, d0[2], d0[3]), ensemble, dt, nsteps)
    Jp2 = minimal_design_objective((d0[1], d0[2] + ϵ, d0[3]), ensemble, dt, nsteps)
    Jm2 = minimal_design_objective((d0[1], d0[2] - ϵ, d0[3]), ensemble, dt, nsteps)
    Jp3 = minimal_design_objective((d0[1], d0[2], d0[3] + ϵ), ensemble, dt, nsteps)
    Jm3 = minimal_design_objective((d0[1], d0[2], d0[3] - ϵ), ensemble, dt, nsteps)
    dJ_dd_fd = ((Jp1 - Jm1) / (2ϵ), (Jp2 - Jm2) / (2ϵ), (Jp3 - Jm3) / (2ϵ))
    loss, dJ_dd = loss_and_gradient(d0, ensemble, dt, nsteps)
    diag = design_diagnostics(d0, ensemble, dt, nsteps)

    @info "Lorenz-63 design diagnostics" diag...
    @info "Gradient wrt design vector d (dJ/dd)" dJ_dd dJ_dd_fd

    # Tight checks: loss from helper should equal direct objective.
    @test loss ≈ J rtol = 1e-12
    @test diag.J ≈ J rtol = 1e-12
    @test all(isfinite, dJ_dd)
    @test dJ_dd[1] ≈ dJ_dd_fd[1] rtol = 5e-4
    @test dJ_dd[2] ≈ dJ_dd_fd[2] rtol = 5e-4
    @test dJ_dd[3] ≈ dJ_dd_fd[3] rtol = 5e-4

    # Tighten behavior check: objective should vary with design.
    J_left  = minimal_design_objective((d0[1] - 0.15, d0[2], d0[3]), ensemble, dt, nsteps)
    J_right = minimal_design_objective((d0[1] + 0.15, d0[2], d0[3]), ensemble, dt, nsteps)
    @test !(J_left ≈ J_right)

    # Basic Adam optimization on sensor design d.
    opt = optimize_design_adam(d0, ensemble, dt, nsteps; η = 1e-1, niter = 300)
    @info "Adam summary" d0 opt.d_final initial_loss=opt.loss_history[1] final_loss=opt.loss_history[end]

    @test all(isfinite, opt.d_final)
    @test opt.loss_history[end] < opt.loss_history[1]

    # Plot loss trajectory.
    fig = plot_loss_history(opt.loss_history; optimizer_name = "Adam")
    plot_path = joinpath(@__DIR__, "lorenz63_design_loss.png")
    save(plot_path, fig)
    @info "Saved gradient-descent loss plot" plot_path
end
