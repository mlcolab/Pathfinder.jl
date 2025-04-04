module PathfinderTuringExt

using Accessors: Accessors
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
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
    else
        prob = DynamicPPL.LogDensityFunction(
            model, varinfo, DynamicPPL.DefaultContext(); adtype
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
        return Turing.Inference.getparams(model, DynamicPPL.unflatten(varinfo, draw))
    end
    param_con_names = map(first, first(draw_con_varinfos))
    draws_con = reduce(
        vcat, Iterators.map(transpose ∘ Base.Fix1(map, last), draw_con_varinfos)
    )
    chns = MCMCChains.Chains(draws_con, param_con_names)
    return chns
end

function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.pathfinder(
        log_density_problem; input=model, adtype=new_adtype, kwargs...
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
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.multipathfinder(
        log_density_problem, ndraws; input=model, adtype=new_adtype, kwargs...
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

    result_new = Accessors.@set (Accessors.@set result.draws_transformed =
        chains).pathfinder_results = single_path_results_new
    return result_new
end

end  # module
