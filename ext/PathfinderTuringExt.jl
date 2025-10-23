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
    draws_to_chains(chain_type, model::DynamicPPL.Model, draws) -> ::chain_type

Convert a `(nparams, ndraws)` matrix of unconstrained `draws` to a
chains object with corresponding constrained draws and names
according to `model`.
"""
function draws_to_chains(chain_type, model::DynamicPPL.Model, draws::AbstractMatrix)
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    params = map(eachcol(draws)) do draw
        draw_varinfo = DynamicPPL.unflatten(varinfo, draw)
        return DynamicPPL.ParamsWithStats(draw_varinfo, model)
    end
    chns = DynamicPPL.to_chains(chain_type, params)
    return chns
end

function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    chain_type=MCMCChains.Chains,
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.pathfinder(
        log_density_problem; input=model, adtype=new_adtype, kwargs...
    )

    # add transformed draws as Chains
    chains = draws_to_chains(chain_type, model, result.draws)
    result_new = Accessors.@set result.draws_transformed = chains
    return result_new
end

function Pathfinder.multipathfinder(
    model::DynamicPPL.Model,
    ndraws::Int;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    chain_type=MCMCChains.Chains,
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.multipathfinder(
        log_density_problem, ndraws; input=model, adtype=new_adtype, kwargs...
    )

    # add transformed draws as Chains
    chains = draws_to_chains(chain_type, model, result.draws)

    # add transformed draws as Chains for each individual path
    single_path_results_new = map(result.pathfinder_results) do r
        single_chains = draws_to_chains(chain_type, model, r.draws)
        r_new = Accessors.@set r.draws_transformed = single_chains
        return r_new
    end

    result_new = Accessors.@set (Accessors.@set result.draws_transformed = chains).pathfinder_results =
        single_path_results_new
    return result_new
end

end  # module
