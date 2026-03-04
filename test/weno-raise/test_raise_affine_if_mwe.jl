"""
Minimal MWE for the Reactant raise=true failure.

Root cause: inside a 3D KA kernel launched via Oceananigans `launch!`, a simple
index-dependent `ifelse` on the x-dimension produces an `affine.if -> i1` block
inside the kernel's bounds-check `affine.if`. The raise pass rejects this.

This file starts from a minimal Oceananigans kernel (Phase A), then strips
Oceananigans away entirely (Phase B) to isolate the pure Reactant pattern.
"""

using Reactant
using KernelAbstractions: @kernel, @index, get_backend, synchronize

# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Minimal Oceananigans reproduction
#   Uses real Oceananigans grids/fields + launch!, but NO advection math.
#   Just: ifelse(i <= 2, field[i,j,k], field[i+1,j,k])
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, XFaceField
using Oceananigans.Grids: Bounded, Periodic, architecture
using Oceananigans.Utils: launch!

@kernel function kernel_oceananigans_ifelse!(out, c, Nx)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i <= Nx ÷ 2, c[i, j, k], c[i, j, k] + 1.0)
end

function run_oceananigans_ifelse!(out, c, Nx, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, kernel_oceananigans_ifelse!, out, c, Nx)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Pure Reactant reproduction — NO Oceananigans
#   3D KA kernel with the SAME StaticSize launch pattern that Oceananigans
#   uses, so that the 3D index decomposition creates affine.if blocks.
# ═══════════════════════════════════════════════════════════════════════════

using KernelAbstractions: NDRange
using KernelAbstractions.NDIteration: StaticSize

@kernel function kernel_pure_3d_ifelse!(out, x, ::Val{Nx}) where Nx
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i <= Nx ÷ 2, x[i, j, k], x[i, j, k] + 1.0)
end

function run_pure_3d!(kernel!, out, x, vNx)
    backend = get_backend(out)
    Nx = typeof(vNx).parameters[1]
    workgroup = (16, 16)
    ndrange = (Nx, Nx, Nx)
    kernel!(backend, workgroup, ndrange)(out, x, vNx)
    synchronize(backend)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Pure Reactant, 1D kernel (control — should pass)
# ═══════════════════════════════════════════════════════════════════════════

@kernel function kernel_pure_1d_ifelse!(out, x, ::Val{N}) where N
    i = @index(Global, Linear)
    @inbounds out[i] = ifelse(i <= N ÷ 2, x[i], x[i] + 1.0)
end

function run_pure_1d!(kernel!, out, x, vN)
    backend = get_backend(out)
    N = typeof(vN).parameters[1]
    kernel!(backend, 16, N)(out, x, vN)
    synchronize(backend)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Probe helper
# ═══════════════════════════════════════════════════════════════════════════

function probe(label, run_fn!, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        @info "  → PASS" label
        return (ok=true, error="")
    catch err
        buf = IOBuffer(); showerror(buf, err)
        msg = String(take!(buf))
        @info "  → FAIL" label
        return (ok=false, error=msg)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

Nx = 4

# Phase A: Oceananigans grid + launch!
@info "═══ Phase A: Oceananigans grid + launch! ═══"
arch = ReactantState()
grid = RectilinearGrid(arch; size=(Nx, Nx, Nx), halo=(3, 3, 3), extent=(1, 1, 1),
                       topology=(Bounded, Periodic, Bounded))
c_field   = CenterField(grid)
out_field = XFaceField(grid)
set!(c_field, (x, y, z) -> x + y)

a_result = probe("Phase A: Oceananigans 3D ifelse",
                 run_oceananigans_ifelse!, out_field, c_field, Nx, grid)

# Phase B: Pure Reactant 3D
@info "═══ Phase B: Pure Reactant 3D (StaticSize workgroup) ═══"
x3d   = Reactant.to_rarray(randn(Nx, Nx, Nx))
out3d = Reactant.to_rarray(zeros(Nx, Nx, Nx))

b_result = probe("Phase B: Pure Reactant 3D ifelse",
                 run_pure_3d!, kernel_pure_3d_ifelse!, out3d, x3d, Val(Nx))

# Phase C: Pure Reactant 1D (control)
@info "═══ Phase C: Pure Reactant 1D (control) ═══"
x1d   = Reactant.to_rarray(randn(Nx))
out1d = Reactant.to_rarray(zeros(Nx))

c_result = probe("Phase C: Pure Reactant 1D ifelse (control)",
                 run_pure_1d!, kernel_pure_1d_ifelse!, out1d, x1d, Val(Nx))

# Results
@info "═══ Results ═══"
for (name, result) in [
    ("Phase A: Oceananigans 3D  ", a_result),
    ("Phase B: Pure Reactant 3D ", b_result),
    ("Phase C: Pure Reactant 1D ", c_result),
]
    status = result.ok ? "PASS" : "FAIL"
    @info "$name : $status"
    if !result.ok
        errmsg = first(result.error, 400)
        @info "  Error (truncated): $errmsg"
    end
end

@info """

Expected:
  Phase A (Oceananigans 3D): FAIL — reproduces the bug
  Phase B (Pure 3D):         FAIL — if the 3D index decomposition is the trigger
  Phase C (Pure 1D):         PASS — no mod-based index, no affine.if -> i1

If B fails, we have a pure Reactant MWE with no Oceananigans dependency.
If B passes but A fails, the trigger is specific to Oceananigans' launch! or OffsetArray.
"""
