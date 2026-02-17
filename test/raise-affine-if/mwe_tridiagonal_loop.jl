using Reactant, KernelAbstractions

Reactant.set_default_backend("cpu")

const ReactantBackend = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend

# Faithful reproduction of solve_batched_tridiagonal_system_kernel! (ZDirection).
# Key structural elements that differ from a naive version:
#   1. Mixed 1D (a, c: size Nk-1) and 3D (b, f, ϕ, t: size Ni×Nj×Nk) coefficient access
#      → creates mixed memref.load (1D) + affine.load (3D) inside scf.while inside affine.if
#   2. Loop bound from size(grid_size_arg) rather than size(ϕ)
#   3. abs(β) > eps(...) branch with ifelse (accumulator-dependent control flow)
@kernel function _thomas_mixed!(ϕ, a, b, c, f, t, Nk)
    i, j = @index(Global, NTuple)
    @inbounds begin
        β  = b[i, j, 1]
        ϕ[i, j, 1] = f[i, j, 1] / β

        for k = 2:Nk
            t[i, j, k] = c[k-1] / β                     # 1D access: c[k-1]
            β = b[i, j, k] - a[k-1] * t[i, j, k]        # 1D: a[k-1], 3D: b[i,j,k]
            fk = f[i, j, k]                              # 3D access
            diag_dom = abs(β) > 10 * eps(Float64)
            ϕ★ = (fk - a[k-1] * ϕ[i, j, k-1]) / β
            ϕ[i, j, k] = ifelse(diag_dom, ϕ★, ϕ[i, j, k])
        end

        for k = Nk-1:-1:1
            ϕ[i, j, k] -= t[i, j, k+1] * ϕ[i, j, k+1]
        end
    end
end

function go!(ϕ, a, b, c, f, t)
    Nk = size(ϕ, 3)
    _thomas_mixed!(ReactantBackend(), (16, 16))(ϕ, a, b, c, f, t, Nk; ndrange=(5, 5))
    return nothing
end

Ni, Nj, Nk = 5, 5, 4
ϕ = Reactant.to_rarray(zeros(Ni, Nj, Nk))
b = Reactant.to_rarray(rand(Ni, Nj, Nk) .+ 2.0)   # 3D diagonal
f = Reactant.to_rarray(rand(Ni, Nj, Nk))            # 3D rhs
t = Reactant.to_rarray(zeros(Ni, Nj, Nk))           # 3D scratch
a = Reactant.to_rarray(ones(Nk - 1))                # 1D lower diagonal (size 3)
c = Reactant.to_rarray(ones(Nk - 1))                # 1D upper diagonal (size 3)

println("Compiling with raise=true...")
compiled! = Reactant.@compile raise=true sync=true go!(ϕ, a, b, c, f, t)
compiled!(ϕ, a, b, c, f, t)
println("Success: ", Array(ϕ)[1,1,:])
