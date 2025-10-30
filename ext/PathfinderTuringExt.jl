module PathfinderTuringExt

using Accessors: Accessors
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
using Random: Random
using Turing: Turing

"""
    create_log_density_problem(model::DynamicPPL.Model, adtype::ADTypes.AbstractADType)

Create a log density problem from a `model`.

The return value is an object implementing the LogDensityProblems API whose log-density is
that of the `model` transformed to unconstrained space with the appropriate log-density
adjustment due to change of variables.
"""
function create_log_density_problem(model::DynamicPPL.Model, adtype::ADTypes.AbstractADType)
    # create an unconstrained VarInfo
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    # DefaultContext ensures that the log-density adjustment is computed
    @static if pkgversion(DynamicPPL) < v"0.35.0"
        prob = DynamicPPL.LogDensityFunction(model, varinfo, DynamicPPL.DefaultContext())
    elseif pkgversion(DynamicPPL) < v"0.37.0"
        prob = DynamicPPL.LogDensityFunction(
            model, varinfo, DynamicPPL.DefaultContext(); adtype
        )
    else
        prob = DynamicPPL.LogDensityFunction(
            model, DynamicPPL.getlogjoint_internal, varinfo; adtype
        )
    end
    return prob
end

function _adtype(prob::DynamicPPL.LogDensityFunction, adtype::ADTypes.AbstractADType)
    hasfield(typeof(prob), :adtype) || return adtype
    return prob.adtype
end

"""
    draws_to_chains(model::DynamicPPL.Model, draws) -> MCMCChains.Chains

Convert a `(nparams, ndraws)` matrix of unconstrained `draws` to a
[`MCMCChains.Chains`](@extref) object with corresponding constrained draws and names
according to `model`.
"""
function draws_to_chains(model::DynamicPPL.Model, draws::AbstractMatrix)
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    draw_con_varinfos = map(eachcol(draws)) do draw
        # this re-evaluates the model but allows supporting dynamic bijectors
        # https://github.com/TuringLang/Turing.jl/issues/2195
        draw_varinfo = DynamicPPL.unflatten(varinfo, draw)
        unlinked_params = DynamicPPL.values_as_in_model(model, true, draw_varinfo)
        iters = map(
            DynamicPPL.varname_and_value_leaves,
            keys(unlinked_params),
            values(unlinked_params),
        )
        return mapreduce(collect, vcat, iters)
    end
    param_con_names = map(first, first(draw_con_varinfos))
    draws_con = reduce(
        vcat, Iterators.map(transpose ∘ Base.Fix1(map, last), draw_con_varinfos)
    )
    chns = MCMCChains.Chains(draws_con, param_con_names)
    return chns
end

@static if isdefined(DynamicPPL, :AbstractInitStrategy)
    struct InitStrategySampler{M<:DynamicPPL.Model,S<:DynamicPPL.AbstractInitStrategy}
        model::M
        strategy::S
    end
    function (spl::InitStrategySampler)(rng::Random.AbstractRNG, point::AbstractVector)
        (; model, strategy) = spl
        varinfo = DynamicPPL.VarInfo(rng, model, strategy)
        varinfo_linked = DynamicPPL.link(varinfo, model)
        copyto!(point, varinfo_linked[:])
    end

    function _maybe_add_sampler_to_kwargs(
        model::DynamicPPL.Model; kwargs...
    )
        # TODO: Change to `InitFromPrior()` (breaking)
        haskey(kwargs, :init_sampler) || return kwargs
        init_sampler = kwargs[:init_sampler]
        if init_sampler isa DynamicPPL.AbstractInitStrategy
            return _merge_keywords(
                kwargs; init_sampler=InitStrategySampler(model, init_sampler)
            )
        else
            return _merge_keywords(kwargs; init_sampler)
        end
    end

    function _format_init(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        init::DynamicPPL.AbstractInitStrategy,
    )
        varinfo = DynamicPPL.VarInfo(rng, model, init)
        varinfo_linked = DynamicPPL.link(varinfo, model)
        return varinfo_linked[:]
    end
    function _format_init(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        init::AbstractVector{<:DynamicPPL.AbstractInitStrategy},
    )
        return map(x -> _format_init(rng, model, x), init)
    end
else
    _maybe_add_sampler_to_kwargs(model::DynamicPPL.Model; kwargs...) = kwargs
end

_format_init(rng::Random.AbstractRNG, model::DynamicPPL.Model, init) = init

function _maybe_add_init_to_kwargs(
    rng::Random.AbstractRNG, model::DynamicPPL.Model; kwargs...
)
    haskey(kwargs, :init) || return kwargs
    return _merge_keywords(kwargs; init=_format_init(rng, model, kwargs[:init]))
end

function _update_kwargs(rng::Random.AbstractRNG, model::DynamicPPL.Model; kwargs...)
    return _maybe_add_init_to_kwargs(
        rng, model; _maybe_add_sampler_to_kwargs(model; kwargs...)...
    )
end

@inline function _merge_keywords(kwargs::Base.Pairs; new_kwargs...)
    return pairs(merge(values(kwargs), values(new_kwargs)))
end

function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.pathfinder(
        log_density_problem;
        input=model,
        adtype=new_adtype,
        rng,
        _update_kwargs(rng, model; kwargs...)...,
    )

    # add transformed draws as Chains
    chains_info = (; pathfinder_result=result)
    chains = Accessors.@set draws_to_chains(model, result.draws).info = chains_info
    result_new = Accessors.@set result.draws_transformed = chains
    return result_new
end

function Pathfinder.multipathfinder(
    model::DynamicPPL.Model,
    ndraws::Int;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.multipathfinder(
        log_density_problem,
        ndraws;
        input=model,
        adtype=new_adtype,
        rng,
        _update_kwargs(rng, model; kwargs...)...,
    )

    # add transformed draws as Chains
    chains_info = (; pathfinder_result=result)
    chains = Accessors.@set draws_to_chains(model, result.draws).info = chains_info

    # add transformed draws as Chains for each individual path
    single_path_results_new = map(result.pathfinder_results) do r
        single_chains_info = (; pathfinder_result=r)
        single_chains = Accessors.@set draws_to_chains(model, r.draws).info =
            single_chains_info
        r_new = Accessors.@set r.draws_transformed = single_chains
        return r_new
    end

    result_new = Accessors.@set (Accessors.@set result.draws_transformed = chains).pathfinder_results =
        single_path_results_new
    return result_new
end

end  # module
