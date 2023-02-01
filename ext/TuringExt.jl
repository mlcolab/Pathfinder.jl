module TuringExt

using Accessors: Accessors
using Random: Random
if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using MCMCChains: MCMCChains
    using Pathfinder: Pathfinder
    using Turing: Turing
else  # using Requires
    using ..DynamicPPL: DynamicPPL
    using ..MCMCChains: MCMCChains
    using ..Pathfinder: Pathfinder
    using ..Turing: Turing
end

# utilities for working with Turing model parameter names using only the DynamicPPL API

"""
    flattened_varnames_list(model::DynamicPPL.Model) -> Vector{Symbol}

Get a vector of varnames as `Symbol`s with one-to-one correspondence to the
flattened parameter vector.

```julia
julia> @model function demo()
           s ~ Dirac(1)
           x = Matrix{Float64}(undef, 2, 4)
           x[1, 1] ~ Dirac(2)
           x[2, 1] ~ Dirac(3)
           x[3] ~ Dirac(4)
           y ~ Dirac(5)
           x[4] ~ Dirac(6)
           x[:, 3] ~ arraydist([Dirac(7), Dirac(8)])
           x[[2, 1], 4] ~ arraydist([Dirac(9), Dirac(10)])
           return s, x, y
       end
demo (generic function with 2 methods)

julia> flattened_varnames_list(demo())
10-element Vector{Symbol}:
 :s
 Symbol("x[1,1]")
 Symbol("x[2,1]")
 Symbol("x[3]")
 Symbol("x[4]")
 Symbol("x[:,3][1]")
 Symbol("x[:,3][2]")
 Symbol("x[[2, 1],4][1]")
 Symbol("x[[2, 1],4][2]")
 :y
```
"""
function flattened_varnames_list(model::DynamicPPL.Model)
    varnames_ranges = varnames_to_ranges(model)
    nsyms = maximum(maximum, values(varnames_ranges))
    syms = Vector{Symbol}(undef, nsyms)
    for (var_name, range) in varnames_to_ranges(model)
        sym = Symbol(var_name)
        if length(range) == 1
            syms[range[begin]] = sym
            continue
        end
        for i in eachindex(range)
            syms[range[i]] = Symbol("$sym[$i]")
        end
    end
    return syms
end

# code snippet shared by @torfjelde
"""
    varnames_to_ranges(model::DynamicPPL.Model)
    varnames_to_ranges(model::DynamicPPL.VarInfo)
    varnames_to_ranges(model::DynamicPPL.Metadata)

Get `Dict` mapping variable names in model to their ranges in a corresponding parameter vector.

# Examples

```julia
julia> @model function demo()
           s ~ Dirac(1)
           x = Matrix{Float64}(undef, 2, 4)
           x[1, 1] ~ Dirac(2)
           x[2, 1] ~ Dirac(3)
           x[3] ~ Dirac(4)
           y ~ Dirac(5)
           x[4] ~ Dirac(6)
           x[:, 3] ~ arraydist([Dirac(7), Dirac(8)])
           x[[2, 1], 4] ~ arraydist([Dirac(9), Dirac(10)])
           return s, x, y
       end
demo (generic function with 2 methods)

julia> demo()()
(1, Any[2.0 4.0 7 10; 3.0 6.0 8 9], 5)

julia> varnames_to_ranges(demo())
Dict{AbstractPPL.VarName, UnitRange{Int64}} with 8 entries:
  s           => 1:1
  x[4]        => 5:5
  x[:,3]      => 6:7
  x[1,1]      => 2:2
  x[2,1]      => 3:3
  x[[2, 1],4] => 8:9
  x[3]        => 4:4
  y           => 10:10
```
"""
function varnames_to_ranges end

varnames_to_ranges(model::DynamicPPL.Model) = varnames_to_ranges(DynamicPPL.VarInfo(model))
function varnames_to_ranges(varinfo::DynamicPPL.UntypedVarInfo)
    return varnames_to_ranges(varinfo.metadata)
end
function varnames_to_ranges(varinfo::DynamicPPL.TypedVarInfo)
    offset = 0
    dicts = map(varinfo.metadata) do md
        vns2ranges = varnames_to_ranges(md)
        vals = collect(values(vns2ranges))
        vals_offset = map(r -> offset .+ r, vals)
        offset += reduce((curr, r) -> max(curr, r[end]), vals; init=0)
        Dict(zip(keys(vns2ranges), vals_offset))
    end

    return reduce(merge, dicts)
end
function varnames_to_ranges(metadata::DynamicPPL.Metadata)
    idcs = map(Base.Fix1(getindex, metadata.idcs), metadata.vns)
    ranges = metadata.ranges[idcs]
    return Dict(zip(metadata.vns, ranges))
end

function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    rng=Random.GLOBAL_RNG,
    init_scale=2,
    init_sampler=Pathfinder.UniformSampler(init_scale),
    init=nothing,
    kwargs...,
)
    var_names = flattened_varnames_list(model)
    prob = Turing.optim_problem(model, Turing.MAP(); constrained=false, init_theta=init)
    init_sampler(rng, prob.prob.u0)
    result = Pathfinder.pathfinder(prob.prob; rng, input=model, kwargs...)
    draws = reduce(vcat, transpose.(prob.transform.(eachcol(result.draws))))
    chns = MCMCChains.Chains(draws, var_names; info=(; pathfinder_result=result))
    result_new = Accessors.@set result.draws_transformed = chns
    return result_new
end

function Pathfinder.multipathfinder(
    model::DynamicPPL.Model,
    ndraws::Int;
    rng=Random.GLOBAL_RNG,
    init_scale=2,
    init_sampler=Pathfinder.UniformSampler(init_scale),
    nruns::Int,
    kwargs...,
)
    var_names = flattened_varnames_list(model)
    fun = Turing.optim_function(model, Turing.MAP(); constrained=false)
    init1 = fun.init()
    init = [init_sampler(rng, init1)]
    for _ in 2:nruns
        push!(init, init_sampler(rng, deepcopy(init1)))
    end
    result = Pathfinder.multipathfinder(fun.func, ndraws; rng, input=model, init, kwargs...)
    draws = reduce(vcat, transpose.(fun.transform.(eachcol(result.draws))))
    chns = MCMCChains.Chains(draws, var_names; info=(; pathfinder_result=result))
    result_new = Accessors.@set result.draws_transformed = chns
    return result_new
end

end  # module
