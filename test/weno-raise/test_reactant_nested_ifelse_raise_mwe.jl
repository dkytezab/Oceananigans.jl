"""
Pure Reactant + KernelAbstractions MWE for the `raise=true` failure.

Root cause traced to `topologically_conditional_interpolation.jl`:
on Bounded grids, high-order advection schemes are wrapped in

    _biased_interpolate_x(i, grid::BoundedX, scheme::HOADV, ...) =
        ifelse(outside_halo(i, Nx, scheme),
               full_interp(i, scheme, ...),
               _biased_interpolate_x(i, grid, scheme.buffer_scheme, ...))

Key detail: `grid.Nx` and `required_halo_size(scheme)` are compile-time constants
(embedded in the grid/scheme structs). So the halo predicate `i >= 4 & i <= 2` is
a pure affine constraint on the traced kernel index. MLIR lowers this to `affine.if`
blocks (not `arith.select`). Nested `affine.if` blocks that yield values inside the
kernel's bounds-check `affine.if` is what Reactant's raise pass cannot handle.

When N is instead a traced runtime argument, MLIR uses `arith.cmpi` + `arith.select`
and the raise pass succeeds — explaining why a naive MWE with traced N passes.

This file tests both: Val{N} (compile-time, like Oceananigans) vs traced N (runtime).
"""

using Reactant
using CUDA
using KernelAbstractions: @kernel, @index, get_backend, synchronize

# ── Stencil interpolations ──

@inline function interp_5pt(x, i)
    @inbounds return 0.1 * x[i-2] + 0.3 * x[i-1] + 0.2 * x[i] + 0.3 * x[i+1] + 0.1 * x[i+2]
end

@inline function interp_3pt(x, i)
    @inbounds return 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1]
end

@inline function interp_2pt(x, i)
    @inbounds return 0.5 * (x[i] + x[i+1])
end

# ── Halo predicates ──
# Mirrors outside_biased_halo_xᶠ(i, Bounded, N, scheme) from Oceananigans.
# Two versions: one with compile-time N (produces affine.if in MLIR),
# one with runtime N (produces arith.cmpi + arith.select).

@inline outside_halo_static(i, ::Val{N}, ::Val{H}) where {N, H} = (i >= H + 1) & (i <= N + 2 - H)
@inline outside_halo_dynamic(i, N, H) = (i >= H + 1) & (i <= N + 2 - H)

# ═══════════════════════════════════════════════════════════════════════
# COMPILE-TIME N (matches Oceananigans: grid.Nx is a constant)
# ═══════════════════════════════════════════════════════════════════════

@kernel function kernel_static_no_ifelse!(out, x, ::Val{N}, ::Val{pad}) where {N, pad}
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = interp_5pt(x, i)
end

@kernel function kernel_static_one_ifelse!(out, x, ::Val{N}, ::Val{pad}) where {N, pad}
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = ifelse(outside_halo_static(gi, Val(N), Val(2)),
                                interp_3pt(x, i),
                                interp_2pt(x, i))
end

@kernel function kernel_static_two_ifelse!(out, x, ::Val{N}, ::Val{pad}) where {N, pad}
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = ifelse(outside_halo_static(gi, Val(N), Val(3)),
                                interp_5pt(x, i),
                                ifelse(outside_halo_static(gi, Val(N), Val(2)),
                                       interp_3pt(x, i),
                                       interp_2pt(x, i)))
end

# ═══════════════════════════════════════════════════════════════════════
# RUNTIME N (naive MWE — N is traced, so MLIR uses arith.select)
# ═══════════════════════════════════════════════════════════════════════

@kernel function kernel_dynamic_no_ifelse!(out, x, N, ::Val{pad}) where pad
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = interp_5pt(x, i)
end

@kernel function kernel_dynamic_one_ifelse!(out, x, N, ::Val{pad}) where pad
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = ifelse(outside_halo_dynamic(gi, N, 2),
                                interp_3pt(x, i),
                                interp_2pt(x, i))
end

@kernel function kernel_dynamic_two_ifelse!(out, x, N, ::Val{pad}) where pad
    gi = @index(Global, Linear)
    i = gi + pad
    @inbounds out[gi] = ifelse(outside_halo_dynamic(gi, N, 2),
                                interp_5pt(x, i),
                                ifelse(outside_halo_dynamic(gi, N, 2),
                                       interp_3pt(x, i),
                                       interp_2pt(x, i)))
end

# ── Launchers ──

function run_static_kernel!(kernel!, out, x, vN, vpad)
    backend = get_backend(out)
    N = typeof(vN).parameters[1]
    kernel!(backend, 16)(out, x, vN, vpad; ndrange=N)
    synchronize(backend)
    return nothing
end

function run_dynamic_kernel!(kernel!, out, x, N, vpad)
    backend = get_backend(out)
    kernel!(backend, 16)(out, x, N, vpad; ndrange=N)
    synchronize(backend)
    return nothing
end

# ── Probe helpers ──

function probe_static(label, kernel!, N, pad)
    x   = Reactant.to_rarray(randn(N + 2pad))
    out = Reactant.to_rarray(zeros(N))
    vN   = Val(N)
    vpad = Val(pad)

    @info "Probing (static N)" label N pad
    try
        f! = @compile raise=true raise_first=true sync=true run_static_kernel!(kernel!, out, x, vN, vpad)
        f!(kernel!, out, x, vN, vpad)
        @info "  → PASS" label
        return (ok=true, error="")
    catch err
        buf = IOBuffer(); showerror(buf, err)
        @info "  → FAIL" label
        return (ok=false, error=String(take!(buf)))
    end
end

function probe_dynamic(label, kernel!, N, pad)
    x   = Reactant.to_rarray(randn(N + 2pad))
    out = Reactant.to_rarray(zeros(N))
    vpad = Val(pad)

    @info "Probing (dynamic N)" label N pad
    try
        f! = @compile raise=true raise_first=true sync=true run_dynamic_kernel!(kernel!, out, x, N, vpad)
        f!(kernel!, out, x, N, vpad)
        @info "  → PASS" label
        return (ok=true, error="")
    catch err
        buf = IOBuffer(); showerror(buf, err)
        @info "  → FAIL" label
        return (ok=false, error=String(take!(buf)))
    end
end

# ── Run ──

N = 16
pad = 3

@info "═══ Static N (compile-time constant, like Oceananigans grid.Nx) ═══"
s_none = probe_static("static: no ifelse (periodic)",         kernel_static_no_ifelse!,  N, pad)
s_one  = probe_static("static: 1-level ifelse (WENO3-bdd)",  kernel_static_one_ifelse!, N, pad)
s_two  = probe_static("static: 2-level ifelse (WENO5-bdd)",  kernel_static_two_ifelse!, N, pad)

@info "═══ Dynamic N (traced runtime value) ═══"
d_none = probe_dynamic("dynamic: no ifelse (periodic)",        kernel_dynamic_no_ifelse!,  N, pad)
d_one  = probe_dynamic("dynamic: 1-level ifelse (WENO3-bdd)", kernel_dynamic_one_ifelse!, N, pad)
d_two  = probe_dynamic("dynamic: 2-level ifelse (WENO5-bdd)", kernel_dynamic_two_ifelse!, N, pad)

@info "═══ Results ═══"
for (name, result) in [
    ("Static  no-ifelse  (periodic) ", s_none),
    ("Static  1-ifelse   (WENO3-bdd)", s_one),
    ("Static  2-ifelse   (WENO5-bdd)", s_two),
    ("Dynamic no-ifelse  (periodic) ", d_none),
    ("Dynamic 1-ifelse   (WENO3-bdd)", d_one),
    ("Dynamic 2-ifelse   (WENO5-bdd)", d_two),
]
    status = result.ok ? "PASS" : "FAIL"
    @info "$name : $status"
    if !result.ok
        @info "  Error:\n$(result.error)"
    end
end

@info """

Expected results:
  Static  (compile-time N → affine.if in MLIR):
    no-ifelse:  PASS
    1-ifelse:   PASS (or FAIL)
    2-ifelse:   FAIL  ← matches Oceananigans WENO5+Bounded failure
  Dynamic (traced N → arith.select in MLIR):
    no-ifelse:  PASS
    1-ifelse:   PASS
    2-ifelse:   PASS  ← no nested affine.if, raise succeeds

If static 2-ifelse fails but dynamic 2-ifelse passes, this confirms
the root cause: compile-time halo bounds → affine.if → nested yields
→ raise pass failure. This is a Reactant limitation, not Oceananigans.
"""
