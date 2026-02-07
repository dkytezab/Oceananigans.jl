# MWE: Proposed fix for Issue 1 — explicit Complex() conversion before store
# This should compile successfully if the issue is just the implicit Float64→ComplexF64 promotion.

using Reactant, KernelAbstractions, CUDA

Reactant.set_default_backend("cpu")

A = Reactant.to_rarray(zeros(ComplexF64, 2, 2))

@kernel function store_real_fixed!(A)
    i, j = @index(Global, NTuple)
    val = Float64(i + j)
    @inbounds A[i, j] = Complex(val, zero(val))  # explicit ComplexF64
end

function go!(A)
    store_real_fixed!(KernelAbstractions.get_backend(A))(A; ndrange=size(A))
end

Reactant.@compile raise=true sync=true go!(A)
