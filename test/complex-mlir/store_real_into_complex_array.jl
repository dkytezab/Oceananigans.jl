# MWE: Float64 store into ComplexF64 array in a KA kernel under Reactant
# Expected error: 'affine.store' op value to store must have the same type as memref element type

using Reactant, KernelAbstractions, CUDA

Reactant.set_default_backend("cpu")

A = Reactant.to_rarray(zeros(ComplexF64, 2, 2))

@kernel function store_real!(A)
    i, j = @index(Global, NTuple)
    @inbounds A[i, j] = Float64(i + j)  # Float64 → ComplexF64 implicit conversion
end

function go!(A)
    store_real!(KernelAbstractions.get_backend(A))(A; ndrange=size(A))
end

Reactant.@compile raise=true sync=true go!(A)
