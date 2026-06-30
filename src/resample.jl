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
- `ntasks::Int=1`: number of parallel tasks for evaluating the target log density when
    computing importance weights
"""
function resample(
    result::MultiPathfinderResult,
    ndraws::Int;
    rng::Random.AbstractRNG=result.rng,
    replace::Bool=true,
    importance::Bool=true,
    ndraws_per_run::Union{Nothing,Int}=nothing,
    ntasks::Int=1,
)
    draws_per_component, psis_result = _candidate_draws_and_psis_result(
        rng, result, ndraws_per_run
    )
    psis_result_new = if importance
        if psis_result === nothing
            components = Distributions.components(result.fit_distribution)
            _compute_psis_result(result.logp, components, draws_per_component; ntasks)
        else
            psis_result
        end
    else
        nothing
    end
    draws, draw_component_ids = _resample(
        rng, draws_per_component, psis_result_new, ndraws; replace
    )
    return _build_resampled_result(result, draws, draw_component_ids, psis_result_new)
end

"""
    _resample(rng, x, psis_result, ndraws; replace=true) -> (draws, component_ids)

Draw `ndraws` samples from `x`, returning a draw matrix and component id vector.

`x` is a 3D `(dim, ndraws_per_component, ncomponents)` array. The returned `draws` has
size `(dim, ndraws)` and `component_ids` is an integer vector of length `ndraws`.
If `psis_result` is a `PSIS.PSISResult`, samples are weighted by its importance weights;
if `nothing`, samples are drawn uniformly.
"""
function _resample(rng, draws_per_component, psis_result, ndraws; replace=true)
    draws_all = reshape(draws_per_component, size(draws_per_component, 1), :)
    sample_inds = if psis_result === nothing
        StatsBase.sample(rng, axes(draws_all, 2), ndraws; replace)
    else
        pweights = StatsBase.ProbabilityWeights(
            psis_result.weights, one(eltype(psis_result.weights))
        )
        StatsBase.sample(rng, axes(draws_all, 2), pweights, ndraws; replace)
    end
    draws = draws_all[:, sample_inds]
    ndraws_per_component = size(draws_per_component, 2)
    draw_component_ids = cld.(sample_inds, ndraws_per_component)
    return draws, draw_component_ids
end

function _compute_psis_result(logp, components, draws_per_component; ntasks)
    log_ratios = _compute_log_importance_ratios(
        logp, components, draws_per_component, ntasks
    )
    return PSIS.psis(log_ratios)
end

function _compute_log_importance_ratios(logp, components, draws_per_component, ntasks)
    log_densities_fit = stack(
        _maybe_tmap(
            Distributions.logpdf, components, eachslice(draws_per_component; dims=3); ntasks
        ),
    )
    log_densities_target = _maybe_tmap(
        logp, eachslice(draws_per_component; dims=(2, 3)); ntasks
    )
    log_ratios = vec(log_densities_target - log_densities_fit)
    return log_ratios
end

function _candidate_draws_and_psis_result(_, result::MultiPathfinderResult, ::Nothing)
    # Use existing draws
    draws_per_component = stack(map(x -> x.draws, result.pathfinder_results))
    return draws_per_component, result.psis_result
end
function _candidate_draws_and_psis_result(
    rng, result::MultiPathfinderResult, ndraws_per_run::Int
)
    # Generate new draws from each mixture component
    components = Distributions.components(result.fit_distribution)
    draws_per_component = stack(map(c -> rand(rng, c, ndraws_per_run), components))
    return draws_per_component, nothing
end

# Extension point for rebuilding draws_transformed after resampling.
_rebuild_draws_transformed(input, result, new_draws) = new_draws

# Extension point: return the chain type to pass to AbstractMCMC.from_samples.
# Chain-type-specific extensions override this so that from_samples dispatches correctly.
function _chain_type_from_chain end

# Applies sample_inds to draws_all and returns a new MultiPathfinderResult.
function _build_resampled_result(
    result::MultiPathfinderResult, draws, draw_component_ids, psis_result
)
    draws_transformed = _rebuild_draws_transformed(result.input, result, draws)
    return MultiPathfinderResult(
        result.input,
        result.optimizer,
        result.rng,
        result.optim_fun,
        result.logp,
        result.fit_distribution,
        draws,
        draw_component_ids,
        result.fit_distribution_transformed,
        draws_transformed,
        result.pathfinder_results,
        psis_result,
    )
end
