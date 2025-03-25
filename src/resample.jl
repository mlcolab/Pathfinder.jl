"""
    resample(rng, x, log_weights, ndraws) -> (draws, psis_result)
    resample(rng, x, ndraws) -> draws

Draw `ndraws` samples from `x`, with replacement.

If `log_weights` is provided, perform Pareto smoothed importance resampling.
"""
function resample(rng, x, log_ratios, ndraws)
    psis_result = PSIS.psis(log_ratios)
    (; weights) = psis_result
    pweights = StatsBase.ProbabilityWeights(weights, one(eltype(weights)))
    return StatsBase.sample(rng, x, pweights, ndraws; replace=true), psis_result
end
resample(rng, x, ndraws) = StatsBase.sample(rng, x, ndraws; replace=true)
