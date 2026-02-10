using Reactant, KernelAbstractions

Reactant.set_default_backend("cpu")

A = Reactant.to_rarray(zeros(Float64, 4))
B = Reactant.to_rarray(zeros(ComplexF64, 4))

@kernel function write!(A, B)
    i = @index(Global)
    @inbounds A[i] = Float64(B[i])  
end

f_kernel!(A, B) = (write!(KernelAbstractions.get_backend(A))(A, B; ndrange=length(A)); nothing)
Reactant.@compile raise=true sync=true f_kernel!(A, B)