module PathfinderTuringExt

if isdefined(Base, :get_extension)
    using Accessors: Accessors
    using ADTypes: ADTypes
    using DynamicPPL: DynamicPPL
    using MCMCChains: MCMCChains
    using Optimization: Optimization
    using Pathfinder: Pathfinder
    using Random: Random
    using Turing: Turing
    import Pathfinder: flattened_varnames_list
else  # using Requires
    using ..Accessors: Accessors
    using ..ADTypes: ADTypes
    using ..DynamicPPL: DynamicPPL
    using ..MCMCChains: MCMCChains
    using ..Optimization: Optimization
    using ..Pathfinder: Pathfinder
    using ..Random: Random
    using ..Turing: Turing
    import ..Pathfinder: flattened_varnames_list
end

"""
    create_log_density_problem(model::DynamicPPL.Model)

Create a log density problem from a `model`.

The return value is an object implementing the LogDensityProblems API whose log-density is
that of the `model` transformed to unconstrained space with the appropriate log-density
adjustment due to change of variables.
"""
function create_log_density_problem(model::DynamicPPL.Model)
    # create an unconstrained VarInfo
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    # DefaultContext ensures that the log-density adjustment is computed
    prob = DynamicPPL.LogDensityFunction(varinfo, model, DynamicPPL.DefaultContext())
    return prob
end

"""
    draws_to_chains(model::DynamicPPL.Model, draws) -> MCMCChains.Chains

Convert a `(nparams, ndraws)` matrix of unconstrained `draws` to an `MCMCChains.Chains`
object with corresponding constrained draws and names according to `model`.
"""
function draws_to_chains(model::DynamicPPL.Model, draws::AbstractMatrix)
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    draw_con_varinfos = map(eachcol(draws)) do draw
        # this re-evaluates the model but allows supporting dynamic bijectors
        # https://github.com/TuringLang/Turing.jl/issues/2195
        return Turing.Inference.getparams(model, DynamicPPL.unflatten(varinfo, draw))
    end
    param_con_names = map(first, first(draw_con_varinfos))
    draws_con = reduce(
        vcat, Iterators.map(transpose ∘ Base.Fix1(map, last), draw_con_varinfos)
    )
    chns = MCMCChains.Chains(draws_con, param_con_names)
    return chns
end

# utilities for working with Turing model parameter names using only the DynamicPPL API

function Pathfinder.flattened_varnames_list(model::DynamicPPL.Model)
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

"""
    transform_to_constrained(
        p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model
    )

Transform a vector of parameters `p` from unconstrained to constrained space.
"""
function transform_to_constrained(
    p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model
)
    p = copy(p)
    @assert DynamicPPL.istrans(vi)
    vi = DynamicPPL.unflatten(vi, p)
    p .= DynamicPPL.invlink!!(vi, model)[:]
    # Restore the linking, since we mutated vi.
    DynamicPPL.link!!(vi, model)
    return p
end

"""
    set_up_model_optimisation(model::DynamicPPL.Model, init)

Create the necessary pieces for running optimisation on `model`.

Returns
* An `Optimization.OptimizationFunction` that evaluates the log density of the model and its
gradient in the unconstrained space.
* The initial value `init` transformed to unconstrained space.
* A function `transform_result` that transforms the results back to constrained space. It
takes a single vector argument.
"""
function set_up_model_optimisation(model::DynamicPPL.Model, init)
    # The inner context deterimines whether we are solving MAP or MLE.
    inner_context = DynamicPPL.DefaultContext()
    ctx = Turing.Optimisation.OptimizationContext(inner_context)
    log_density = Turing.Optimisation.OptimLogDensity(model, ctx)
    # Initialise the varinfo with the initial value and then transform to unconstrained
    # space.
    Accessors.@set log_density.varinfo = DynamicPPL.unflatten(log_density.varinfo, init)
    transformed_varinfo = DynamicPPL.link(log_density.varinfo, log_density.model)
    log_density = Accessors.@set log_density.varinfo = transformed_varinfo
    init = log_density.varinfo[:]
    # Create a function that applies the appropriate inverse transformation to results, to
    # bring them back to constrained space.
    transform_result(p) = transform_to_constrained(p, log_density.varinfo, model)
    f = Optimization.OptimizationFunction(
        (x, _) -> log_density(x),;
        grad = (G,x,p) -> log_density(nothing, G, x),
    )
    return f, init, transform_result
end

function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    rng=Random.GLOBAL_RNG,
    init_scale=2,
    init_sampler=Pathfinder.UniformSampler(init_scale),
    init=nothing,
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    kwargs...,
)
    # If no initial value is provided, sample from prior.
    init = init === nothing ? rand(Vector, model) : init
    f, init, transform_result = set_up_model_optimisation(model, init)
    prob = Optimization.OptimizationProblem(f, init)
    init_sampler(rng, init)
    result = Pathfinder.pathfinder(prob; rng, input=model, kwargs...)
    chns_info = (; pathfinder_result=result)
    chns = Accessors.@set draws_to_chains(model, result.draws).info = chns_info
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
    adtype=Pathfinder.default_ad(),
    kwargs...,
)
    # Sample from prior.
    init1 = rand(Vector, model)
    fun, init1, transform_result = set_up_model_optimisation(model, init1)
    init = [init_sampler(rng, init1)]
    for _ in 2:nruns
        push!(init, init_sampler(rng, deepcopy(init1)))
    end
    result = Pathfinder.multipathfinder(fun, ndraws; rng, input=model, init, kwargs...)
    chns_info = (; pathfinder_result=result)
    chns = Accessors.@set draws_to_chains(model, result.draws).info = chns_info
    result_new = Accessors.@set result.draws_transformed = chns
    return result_new
end

end  # module
