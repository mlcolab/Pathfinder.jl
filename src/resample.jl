"""
    resample(rng, x[, log_weights], ndraws)

Draw `ndraws` samples from `x`, with replacement.

If `ndraws` is provided, perform Pareto smoothed importance resampling.
"""
function resample(rng, x, log_ratios, ndraws)
    log_weights, _ = PSIS.psis(log_ratios; normalize=true)
    weights = StatsBase.pweights(exp.(log_weights))
    return StatsBase.sample(rng, x, weights, ndraws; replace=true)
end
resample(rng, x, ndraws) = StatsBase.sample(rng, x, ndraws; replace=true)
