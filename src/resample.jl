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
    draws_all, eff_ndraws_per_run, psis_or_ratios = _get_candidate_draws(
        rng, result, ndraws_per_run, importance, ntasks
    )
    sample_inds, new_psis = _resample(
        rng, axes(draws_all, 2), psis_or_ratios, ndraws; replace
    )
    return _build_resampled_result(
        result, draws_all, sample_inds, eff_ndraws_per_run, new_psis
    )
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
    pweights = StatsBase.ProbabilityWeights(
        psis_result.weights, one(eltype(psis_result.weights))
    )
    return StatsBase.sample(rng, x, pweights, ndraws; replace), psis_result
end

function _resample(rng, x, psis_result::PSIS.PSISResult, ndraws; replace=true)
    pweights = StatsBase.ProbabilityWeights(
        psis_result.weights, one(eltype(psis_result.weights))
    )
    return StatsBase.sample(rng, x, pweights, ndraws; replace), psis_result
end

function _resample(rng, x, ::Nothing, ndraws; replace=true)
    return (StatsBase.sample(rng, x, ndraws; replace), nothing)
end

# Shared log-density ratio computation for importance resampling.
# Returns log(p_target(x)) - log(q_component(x)) for each draw, where each draw x from
# component k is evaluated against q_k (not the mixture), matching how multipathfinder
# computes importance weights.
function _compute_log_densities_ratios(logp, pathfinder_results, draws_all, ntasks)
    log_densities_fit = _maybe_tmapreduce(
        x -> Distributions.logpdf(x.fit_distribution, x.draws),
        vcat,
        pathfinder_results,
        ntasks,
    )
    log_densities_target = _maybe_tmap(logp, eachcol(draws_all), ntasks)
    return log_densities_target - log_densities_fit
end

# Use existing draws from pathfinder_results; reuse stored PSIS weights when available.
function _get_candidate_draws(
    rng, result::MultiPathfinderResult, ::Nothing, importance, ntasks
)
    draws_all = reduce(hcat, map(x -> x.draws, result.pathfinder_results))
    eff_n = size(first(result.pathfinder_results).draws, 2)
    if !importance
        return draws_all, eff_n, nothing
    elseif result.psis_result !== nothing
        return draws_all, eff_n, result.psis_result  # reuse stored weights — no logp calls
    else
        return draws_all, eff_n, _compute_log_densities_ratios(
            result.logp, result.pathfinder_results, draws_all, ntasks
        )
    end
end

# Generate fresh draws from each mixture component using rand_and_logpdf for efficiency.
function _get_candidate_draws(
    rng, result::MultiPathfinderResult, ndraws_per_run::Int, importance, ntasks
)
    components = Distributions.components(result.fit_distribution)
    if importance
        draws_and_logq = map(c -> rand_and_logpdf(rng, c, ndraws_per_run), components)
        draws_all = reduce(hcat, map(first, draws_and_logq))
        log_densities_fit = reduce(vcat, map(last, draws_and_logq))
        log_densities_target = _maybe_tmap(result.logp, eachcol(draws_all), ntasks)
        return draws_all, ndraws_per_run, log_densities_target - log_densities_fit
    else
        draws_all = reduce(hcat, map(c -> rand(rng, c, ndraws_per_run), components))
        return draws_all, ndraws_per_run, nothing
    end
end

# Extension point for rebuilding draws_transformed after resampling.
_rebuild_draws_transformed(input, result, new_draws) = new_draws

# Extension point: return the chain type to pass to AbstractMCMC.from_samples.
# Chain-type-specific extensions override this so that from_samples dispatches correctly.
function _chain_type_from_chain end

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
