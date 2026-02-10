# Upstream MWE: ComplexF64 in KA kernels broken under Reactant
#
# Both reading and writing ComplexF64 arrays in KernelAbstractions kernels
# produces invalid MLIR when compiled with raise=true.
# Broadcasts on the same arrays work fine.
#
# File this with Reactant.jl — no Oceananigans dependency needed.

using Reactant, KernelAbstractions

Reactant.set_default_backend("cpu")

A = Reactant.to_rarray(zeros(ComplexF64, 4, 4))
B = Reactant.to_rarray(ones(4, 4))

# ── Writing to ComplexF64 array ──────────────────────────────────
# Error: 'affine.store' op value to store must have the same type as memref element type

@kernel function write_complex!(A, B)
    i, j = @index(Global, NTuple)
    @inbounds A[i, j] = Complex(B[i, j], zero(B[i, j]))
end

function write_ka!(A, B)
    write_complex!(KernelAbstractions.get_backend(A))(A, B; ndrange=size(A))
end

function write_broadcast!(A, B)
    A .= Complex.(B, zero.(B))
end

# ── Reading from ComplexF64 array ────────────────────────────────
# Error: 'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f64>'

@kernel function read_complex!(B, A)
    i, j = @index(Global, NTuple)
    @inbounds B[i, j] = real(A[i, j])
end

function read_ka!(B, A)
    read_complex!(KernelAbstractions.get_backend(A))(B, A; ndrange=size(A))
end

function read_broadcast!(B, A)
    B .= real.(A)
end

# ── Tests ────────────────────────────────────────────────────────

println("write via broadcast:  ", try Reactant.@compile raise=true sync=true write_broadcast!(A, B); "PASS" catch; "FAIL" end)
println("write via KA kernel:  ", try Reactant.@compile raise=true sync=true write_ka!(A, B);        "PASS" catch; "FAIL" end)
println("read via broadcast:   ", try Reactant.@compile raise=true sync=true read_broadcast!(B, A);  "PASS" catch; "FAIL" end)
println("read via KA kernel:   ", try Reactant.@compile raise=true sync=true read_ka!(B, A);         "PASS" catch; "FAIL" end)
