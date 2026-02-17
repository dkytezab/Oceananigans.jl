using Reactant, KernelAbstractions, CUDA

Reactant.set_default_backend("cpu")

const ReactantBackend = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend

@kernel function _add_one!(out, inp)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = inp[i, j] + 1.0
end

function go!(out, inp)
    _add_one!(ReactantBackend(), wg)(out, inp; ndrange=size)
    return nothing
end

size = (4, 4)
wg = (16, 16)

out = Reactant.to_rarray(zeros(size...))
inp = Reactant.to_rarray(ones(size...))

compiled! = Reactant.@compile raise=true sync=true go!(out, inp)
compiled!(out, inp)
