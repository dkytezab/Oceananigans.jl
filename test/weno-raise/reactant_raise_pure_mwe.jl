using Reactant
using OffsetArrays
using KernelAbstractions: @kernel, @index, synchronize, StaticSize

const ReactantKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const ReactantBackend = ReactantKAExt.ReactantBackend

@kernel function my_kernel!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i <= 2, c[i, j, k], c[i, j, k] + 1.0)
end

function run!(out, c)
    my_kernel!(ReactantBackend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c)
    synchronize(ReactantBackend())
    return nothing
end

N = 4; H = 3; S = N + 2H
raw_c   = Reactant.to_rarray(randn(S, S, S))
raw_out = Reactant.to_rarray(zeros(S, S, S))
c   = OffsetArray(raw_c,   -H+1:N+H, -H+1:N+H, -H+1:N+H)
out = OffsetArray(raw_out,  -H+1:N+H, -H+1:N+H, -H+1:N+H)

compiled! = @compile raise=true raise_first=true sync=true run!(out, c)
compiled!(out, c)
println(out)
