"""
    resample(result::MultiPathfinderResult, ndraws; kwargs...) -> MultiPathfinderResult

Resample `ndraws` draws from a fitted [`MultiPathfinderResult`](@ref).

All fields of the result are preserved except `draws`, `draw_component_ids`,
`draws_transformed`, and `psis_result`, which reflect the new draws.

# Keywords
- `rng::AbstractRNG`: pseudorandom number generator (default: `result.rng`)
- `replace::Bool=true`: sample with or without replacement
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
    replace=true,
    importance=true,
    ndraws_per_run=nothing,
    ntasks=1,
)
    draws_all, eff_ndraws_per_run = _get_candidate_draws(rng, result, ndraws_per_run)

    psis_or_ratios = if !importance
        nothing
    elseif ndraws_per_run === nothing && result.psis_result !== nothing
        result.psis_result  # reuse stored weights — avoids re-evaluating logp
    else
        _compute_log_densities_ratios(result, draws_all, ntasks)
    end

    sample_inds, new_psis = _resample(rng, axes(draws_all, 2), psis_or_ratios, ndraws; replace)
    return _build_resampled_result(result, draws_all, sample_inds, eff_ndraws_per_run, new_psis)
end

"""
    _resample(rng, x, log_weights, ndraws; replace=true) -> (draws, psis_result)
    _resample(rng, x, psis_result::PSIS.PSISResult, ndraws; replace=true) -> (draws, psis_result)
    _resample(rng, x, ::Nothing, ndraws; replace=true) -> (draws, nothing)

Draw `ndraws` samples from `x`, returning `(samples, psis_result_or_nothing)`.

- With `log_weights`: perform Pareto smoothed importance resampling.
- With a `PSISResult`: reuse pre-computed PSIS weights.
- With `nothing`: sample uniformly.
"""
function _resample(rng, x, log_ratios, ndraws; replace=true)
    psis_result = PSIS.psis(log_ratios)
    pweights = StatsBase.ProbabilityWeights(psis_result.weights, one(eltype(psis_result.weights)))
    return StatsBase.sample(rng, x, pweights, ndraws; replace), psis_result
end

function _resample(rng, x, psis_result::PSIS.PSISResult, ndraws; replace=true)
    pweights = StatsBase.ProbabilityWeights(psis_result.weights, one(eltype(psis_result.weights)))
    return StatsBase.sample(rng, x, pweights, ndraws; replace), psis_result
end

_resample(rng, x, ::Nothing, ndraws; replace=true) = (StatsBase.sample(rng, x, ndraws; replace), nothing)

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

# Extension point: return the chain constructor to pass to AbstractMCMC.from_samples.
# Chain-type-specific extensions override this so that from_samples dispatches correctly.
# For example, MCMCChains defines from_samples only for the bare unparameterized type, so
# its override returns MCMCChains.Chains rather than the concrete parameterized type.
_chain_constructor(chain) = typeof(chain)

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
