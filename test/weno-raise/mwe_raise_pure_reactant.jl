using Reactant
using KernelAbstractions: @kernel, @index, StaticSize

const RKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const Backend = RKAExt.ReactantBackend

struct Params
    Nx::Int
end

# ── Version A: ifelse (known FAIL) ──

@kernel function kernel_ifelse!(out, p, c)
    i, j, k = @index(Global, NTuple)
    N = p.Nx
    @inbounds out[i, j, k] = ifelse((i >= 4) & (i <= N - 2), c[i, j, k], c[i, j, k] + 1.0)
end

function run_ifelse!(out, p, c)
    kernel_ifelse!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, p, c)
    return nothing
end

# ── Version B: @trace if (does this change the MLIR path?) ──

@kernel function kernel_trace!(out, p, c)
    i, j, k = @index(Global, NTuple)
    N = p.Nx
    @trace if (i >= 4) & (i <= N - 2)
        @inbounds out[i, j, k] = c[i, j, k]
    else
        @inbounds out[i, j, k] = c[i, j, k] + 1.0
    end
end

function run_trace!(out, p, c)
    kernel_trace!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, p, c)
    return nothing
end

out = Reactant.ConcreteRArray(zeros(Float64, 4, 4, 4))
c   = Reactant.ConcreteRArray(rand(Float64, 4, 4, 4))
p   = Params(4)

@info "Version A: ifelse"
try
    f! = @compile raise=true raise_first=true sync=true run_ifelse!(out, p, c)
    f!(out, p, c)
    @info "  → PASS"
catch
    @info "  → FAIL"
end

@info "Version B: @trace if"
try
    f! = @compile raise=true raise_first=true sync=true run_trace!(out, p, c)
    f!(out, p, c)
    @info "  → PASS"
catch
    @info "  → FAIL"
end
