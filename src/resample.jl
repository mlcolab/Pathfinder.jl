"""
    resample(rng, x, log_weights, ndraws; replace=true) -> (draws, psis_result)
    resample(rng, x, ndraws; replace=true) -> draws

Draw `ndraws` samples from `x`.

If `log_weights` is provided, perform Pareto smoothed importance resampling.
"""
function resample(rng, x, log_ratios, ndraws; replace=true)
    psis_result = PSIS.psis(log_ratios)
    (; weights) = psis_result
    pweights = StatsBase.ProbabilityWeights(weights, one(eltype(weights)))
    return StatsBase.sample(rng, x, pweights, ndraws; replace), psis_result
end
resample(rng, x, ndraws; replace=true) = StatsBase.sample(rng, x, ndraws; replace)

# Returns (draws_all, effective_ndraws_per_run).
# When ndraws_per_run is nothing, uses existing draws stored in pathfinder_results.
# When ndraws_per_run is an integer, generates fresh draws from each mixture component.
function _get_candidate_draws(rng, result::MultiPathfinderResult, ndraws_per_run)
    if ndraws_per_run === nothing
        draws_all = mapreduce(x -> x.draws, hcat, result.pathfinder_results)
        return draws_all, size(result.pathfinder_results[1].draws, 2)
    else
        components = Distributions.components(result.fit_distribution)
        draws_all = mapreduce(c -> rand(rng, c, ndraws_per_run), hcat, components)
        return draws_all, ndraws_per_run
    end
end

# Computes log(p_target(x)) - log(q_fit(x)) for each column of draws_all.
function _compute_log_densities_ratios(result::MultiPathfinderResult, draws_all, ntasks)
    log_densities_fit = _maybe_tmap(
        col -> Distributions.logpdf(result.fit_distribution, col),
        eachcol(draws_all),
        ntasks,
    )
    log_densities_target = _maybe_tmap(result.logp, eachcol(draws_all), ntasks)
    return log_densities_target - log_densities_fit
end

# Extension point for rebuilding draws_transformed after resampling.
# The Turing extension overrides this for DynamicPPL.Model inputs to return a Chains object.
_rebuild_draws_transformed(input, result, new_draws) = new_draws

# Applies sample_inds to draws_all and returns a new MultiPathfinderResult.
function _build_resampled_result(
    result::MultiPathfinderResult, draws_all, sample_inds, ndraws_per_run, new_psis_result
)
    new_draws = draws_all[:, sample_inds]
    new_draw_component_ids = cld.(sample_inds, ndraws_per_run)
    new_draws_transformed = _rebuild_draws_transformed(result.input, result, new_draws)
    return MultiPathfinderResult(
        result.input,
        result.optimizer,
        result.rng,
        result.optim_fun,
        result.logp,
        result.fit_distribution,
        new_draws,
        new_draw_component_ids,
        result.fit_distribution_transformed,
        new_draws_transformed,
        result.pathfinder_results,
        new_psis_result,
    )
end

"""
    resample(result::MultiPathfinderResult, ndraws; kwargs...) -> MultiPathfinderResult

Resample `ndraws` draws from a fitted [`MultiPathfinderResult`](@ref).

All fields of the result are preserved except `draws`, `draw_component_ids`,
`draws_transformed`, and `psis_result`, which reflect the new draws.

# Keywords
- `rng::AbstractRNG`: pseudorandom number generator (default: `result.rng`)
- `replace::Bool=false`: sample with or without replacement
- `importance::Bool=true`: use PSIS-smoothed importance weights
- `ndraws_per_run::Union{Nothing,Int}=nothing`: if set, generate this many fresh draws per
    run from `fit_distribution` before resampling; otherwise reuse existing draws from
    `pathfinder_results`. Setting this is useful when more draws are needed than were
    originally requested.
- `ntasks::Int=1`: number of parallel tasks for log-density evaluation, used only when
    generating fresh draws with `importance=true`
"""
function resample(
    result::MultiPathfinderResult,
    ndraws;
    rng=result.rng,
    replace=false,
    importance=true,
    ndraws_per_run=nothing,
    ntasks=1,
)
    draws_all, eff_ndraws_per_run = _get_candidate_draws(rng, result, ndraws_per_run)

    sample_inds, new_psis = if importance
        if ndraws_per_run === nothing && result.psis_result !== nothing
            # Reuse stored PSIS weights — avoids re-evaluating logp
            pweights = StatsBase.ProbabilityWeights(
                result.psis_result.weights, one(eltype(result.psis_result.weights))
            )
            StatsBase.sample(rng, axes(draws_all, 2), pweights, ndraws; replace),
            result.psis_result
        else
            log_densities_ratios = _compute_log_densities_ratios(result, draws_all, ntasks)
            resample(rng, axes(draws_all, 2), log_densities_ratios, ndraws; replace)
        end
    else
        resample(rng, axes(draws_all, 2), ndraws; replace), nothing
    end

    return _build_resampled_result(result, draws_all, sample_inds, eff_ndraws_per_run, new_psis)
end
