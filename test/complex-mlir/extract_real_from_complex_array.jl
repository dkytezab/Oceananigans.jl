# MWE: real() on ComplexF64 in a KA kernel under Reactant
# Expected error: 'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f64>'

using Reactant, KernelAbstractions, CUDA

Reactant.set_default_backend("cpu")

A = Reactant.to_rarray(ComplexF64[1+2im 3+4im; 5+6im 7+8im])
B = Reactant.to_rarray(zeros(2, 2))

@kernel function extract_real!(B, A)
    i, j = @index(Global, NTuple)
    @inbounds B[i, j] = real(A[i, j])
end

function go!(B, A)
    extract_real!(KernelAbstractions.get_backend(A))(B, A; ndrange=size(A))
end

Reactant.@compile raise=true sync=true go!(B, A)
