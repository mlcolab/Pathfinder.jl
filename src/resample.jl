"""
    resample(rng, x[, log_weights], ndraws)

Draw `ndraws` samples from `x`, with replacement.

If `log_weights` is provided, perform Pareto smoothed importance resampling.
"""
function resample(rng, x, log_ratios, ndraws)
    result = PSIS.psis(log_ratios)
    weights = result.weights
    pweights = StatsBase.ProbabilityWeights(weights, one(eltype(weights)))
    return StatsBase.sample(rng, x, pweights, ndraws; replace=true)
end
resample(rng, x, ndraws) = StatsBase.sample(rng, x, ndraws; replace=true)
