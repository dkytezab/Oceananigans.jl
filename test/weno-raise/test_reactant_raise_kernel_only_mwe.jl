using Reactant
using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
This file is a kernel-only (Reactant + KernelAbstractions) distillation.
It manually mimics the bounded-x branch structure we saw in WENO lowering:
- boundary predicates on index
- nested `if` regions
- multi-value branch yields
- select between high-order-like and fallback reconstructions

Goal: trigger the same `raise=true` pass-manager failure without Oceananigans code.
"""

@inline function _clamp_index(i, N)
    return ifelse(i < 1, 1, ifelse(i > N, N, i))
end

@kernel function kernel_weno_like_branch!(out, x, N)
    i = @index(Global, Linear)
    if i <= N
        im2 = _clamp_index(i - 2, N)
        im1 = _clamp_index(i - 1, N)
        ip1 = _clamp_index(i + 1, N)
        ip2 = _clamp_index(i + 2, N)

        xm2 = @inbounds x[im2]
        xm1 = @inbounds x[im1]
        x0  = @inbounds x[i]
        xp1 = @inbounds x[ip1]
        xp2 = @inbounds x[ip2]

        # Mirrors the "bounded interior/buffer" condition style in the failing IR.
        in_inner = (i >= 2) & (i <= N - 1)
        near_left = (i >= 2) & (i <= 3)

        # Intentional multi-value branch yield.
        βa, βb, βc = if near_left
            (abs(xp1 - x0), abs(x0 - xm1), abs(xm1 - xm2))
        else
            (abs(xp2 - xp1), abs(xp1 - x0), abs(x0 - xm1))
        end

        # Another branch with yielded values, then weighted combination.
        w1, w2, w3 = if in_inner & near_left
            (0.6, 0.3, 0.1)
        else
            (0.5, 0.5, 0.0)
        end

        s = βa + βb + βc + 1e-8
        α1 = w1 / s
        α2 = w2 / s
        α3 = w3 / s

        recon_hi = α1 * xp1 + α2 * x0 + α3 * xm1
        recon_lo = 0.5 * (x0 + xp1)

        @inbounds out[i] = ifelse(in_inner, recon_hi, recon_lo)
    end
end

@kernel function kernel_simple_linear!(out, x, N)
    i = @index(Global, Linear)
    if i <= N
        ip1 = _clamp_index(i + 1, N)
        @inbounds out[i] = 0.5 * (x[i] + x[ip1])
    end
end

function run_kernel!(kernel!, out, x, N)
    backend = get_backend(out)
    launched! = kernel!(backend, 16)
    launched!(out, x, N; ndrange=N)
    synchronize(backend)
    return nothing
end

function probe_raise(name, kernel!, out, x, N)
    @info "Compiling kernel probe" name N
    try
        f! = @compile raise=true raise_first=true sync=true run_kernel!(kernel!, out, x, N)
        @info "Running compiled kernel probe" name
        f!(kernel!, out, x, N)
        return (ok=true, error="")
    catch err
        io = IOBuffer()
        showerror(io, err)
        return (ok=false, error=String(take!(io)))
    end
end

function run_reactant_kernel_only_mwe(; N=256, T=Float64)
    x = Reactant.to_rarray(rand(T, N), track_numbers=Number)
    out = Reactant.to_rarray(zeros(T, N), track_numbers=Number)

    branchy = probe_raise("weno-like-branchy", kernel_weno_like_branch!, out, x, N)
    simple  = probe_raise("simple-linear", kernel_simple_linear!, out, x, N)

    @info "Kernel-only Reactant distillation results" branchy simple
    return (; branchy, simple)
end

# Run on include / direct execution for quick iteration.
run_reactant_kernel_only_mwe()
