#=
MWE: ComplexF64 operations — broadcast vs KA kernel

Tests whether Reactant handles ComplexF64 correctly through broadcasts
but not through KA kernels. This helps determine if the issue is
KA-specific or a general Reactant ComplexF64 problem.
=#

using Reactant, KernelAbstractions, CUDA, Test

Reactant.set_default_backend("cpu")

A_complex = Reactant.to_rarray(zeros(ComplexF64, 2, 2))
B_real    = Reactant.to_rarray(ones(2, 2))

#####
##### Test 1: Broadcast Float64 → ComplexF64 (Issue 1 analog)
#####

function broadcast_store!(A, B)
    A .= Complex.(B, zero.(B))
    return nothing
end

@testset "Broadcast store Float64 → ComplexF64" begin
    @info "Testing broadcast Complex.(B, zero.(B))..."
    try
        compiled = Reactant.@compile raise=true sync=true broadcast_store!(A_complex, B_real)
        compiled(A_complex, B_real)
        @test true
        @info "  PASSED"
    catch e
        @test false
        @info "  FAILED: $e"
    end
end

#####
##### Test 2: Broadcast real() on ComplexF64 (Issue 2 analog)
#####

A_complex2 = Reactant.to_rarray(ComplexF64[1+2im 3+4im; 5+6im 7+8im])
B_real2    = Reactant.to_rarray(zeros(2, 2))

function broadcast_extract!(B, A)
    B .= real.(A)
    return nothing
end

@testset "Broadcast real.(ComplexF64)" begin
    @info "Testing broadcast real.(A)..."
    try
        compiled = Reactant.@compile raise=true sync=true broadcast_extract!(B_real2, A_complex2)
        compiled(B_real2, A_complex2)
        @test true
        @info "  PASSED"
    catch e
        @test false
        @info "  FAILED: $e"
    end
end

#####
##### Test 3: KA kernel Float64 → ComplexF64 (Issue 1 — known failure)
#####

@kernel function ka_store!(A, B)
    i, j = @index(Global, NTuple)
    @inbounds A[i, j] = Complex(B[i, j], zero(B[i, j]))
end

function kernel_store!(A, B)
    ka_store!(KernelAbstractions.get_backend(A))(A, B; ndrange=size(A))
end

@testset "KA kernel store Float64 → ComplexF64 (expected failure)" begin
    @info "Testing KA kernel Complex(B[i,j], zero(B[i,j]))..."
    @test_broken begin
        compiled = Reactant.@compile raise=true sync=true kernel_store!(A_complex, B_real)
        compiled(A_complex, B_real)
        true
    end
end

#####
##### Test 4: KA kernel real() on ComplexF64 (Issue 2 — known failure)
#####

@kernel function ka_extract!(B, A)
    i, j = @index(Global, NTuple)
    @inbounds B[i, j] = real(A[i, j])
end

function kernel_extract!(B, A)
    ka_extract!(KernelAbstractions.get_backend(A))(B, A; ndrange=size(A))
end

@testset "KA kernel real() on ComplexF64 (expected failure)" begin
    @info "Testing KA kernel real(A[i,j])..."
    @test_broken begin
        compiled = Reactant.@compile raise=true sync=true kernel_extract!(B_real2, A_complex2)
        compiled(B_real2, A_complex2)
        true
    end
end
