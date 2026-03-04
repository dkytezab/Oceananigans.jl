using Reactant
using KernelAbstractions: @kernel, @index, StaticSize

const RKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const Backend = RKAExt.ReactantBackend

function probe(label, run_fn!, args...)
    @info "Probing" label
    try
        f! = @compile raise=true raise_first=true sync=true run_fn!(args...)
        f!(args...)
        @info "  → PASS" label
    catch
        @info "  → FAIL" label
    end
end

out = Reactant.ConcreteRArray(zeros(Float64, 4, 4, 4))
c   = Reactant.ConcreteRArray(rand(Float64, 4, 4, 4))

# T1: ifelse with a pure literal condition (no traced values in condition)
@kernel function k1!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(true, c[i, j, k], c[i, j, k] + 1.0)
end
function r1!(out, c)
    k1!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c); nothing
end

# T2: ifelse where condition depends on the index (i > 2)
@kernel function k2!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i > 2, c[i, j, k], c[i, j, k] + 1.0)
end
function r2!(out, c)
    k2!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c); nothing
end

# T3: ifelse where condition depends on array value (traced Float64)
@kernel function k3!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(c[i, j, k] > 0.5, c[i, j, k], c[i, j, k] + 1.0)
end
function r3!(out, c)
    k3!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c); nothing
end

# T4: ifelse where condition depends on j (floordiv path, not mod)
@kernel function k4!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(j > 2, c[i, j, k], c[i, j, k] + 1.0)
end
function r4!(out, c)
    k4!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c); nothing
end

# T5: ifelse with i condition AND array read in both branches
@kernel function k5!(out, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ifelse(i > 2, c[i, j, k] * 2.0, c[i, j, k] + 1.0)
end
function r5!(out, c)
    k5!(Backend(), StaticSize((16, 16)), StaticSize((4, 4, 4)))(out, c); nothing
end

probe("T1: ifelse(true, ...)",           r1!, out, c)
probe("T2: ifelse(i > 2, ...)",          r2!, out, c)
probe("T3: ifelse(c[i,j,k] > 0.5, ...)", r3!, out, c)
probe("T4: ifelse(j > 2, ...)",          r4!, out, c)
probe("T5: ifelse(i > 2, load*2, load+1)", r5!, out, c)
