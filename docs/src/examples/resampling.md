# Resampling from a fitted approximation

After running [`multipathfinder`](@ref), the fitted mixture distribution and individual-run draws
are stored in the result.
[`resample`](@ref) lets you draw new samples from this approximation without re-running Pathfinder.
Two common workflows are:

1. **Selecting unique MCMC initialization points**: When the Pareto shape ``k`` is large, the default importance resampling *with replacement* can result in all returned draws being identical.
   Importance resampling *without replacement* yields at most one draw per unique candidate, which is more suitable as distinct starting points for independent MCMC chains.
2. **Requesting more draws**: When the original run used too few `ndraws_per_run`, fresh draws can be generated from the fitted mixture and importance-resampled, without re-fitting, at the cost of additional log-density evaluations.

## Setup

We use the funnel distribution from the [Quickstart](@ref "A 100-dimensional funnel") as our running example.

```@example 1
using ADTypes, Pathfinder, Random, ReverseDiff

Random.seed!(68)

function logp_funnel(x)
    n = length(x)
    τ = x[1]
    β = view(x, 2:n)
    return ((τ / 3)^2 + (n - 1) * τ + sum(b -> abs2(b * exp(-τ / 2)), β)) / -2
end

ndraws = 200
result = multipathfinder(logp_funnel, ndraws; dim=100, nruns=20, init_scale=10, adtype=AutoReverseDiff())
```

The Pareto shape ``k`` diagnostic is very bad, indicating that the importance resampled draws do not represent the target distribution very well.
Sometimes this means that only a small number of distinct draws are proposed:

```@example 1
ndraws_distinct = length(unique(eachcol(result.draws)))
ndraws_distinct / ndraws  # fraction of draws that are distinct
```

## Workflow 1: unique MCMC initialization points

To seed ``N`` independent MCMC chains, we need ``N`` distinct starting points.
`resample` with `replace=false` selects unique draws using the stored PSIS weights, with no additional log-density evaluations.

```@example 1
nchains = 8
init_result = resample(result, nchains; replace=false)
```

!!! note "Interpreting draws when sampling without replacement"
    Sampling draws without replacement results in any estimates computed from those draws being biased.
    As a result, one should probably only use these draws for initializing sampling algorithms.

Each column of `init_result.draws` is a distinct point from the original candidate pool:

```@example 1
length(unique(eachcol(init_result.draws))) / nchains
```

The resampled result preserves the fitted distribution and all other fields, so it can be passed directly to any downstream Pathfinder-aware workflow:

```@example 1
init_result.fit_distribution === result.fit_distribution
```

## Workflow 2: more draws from the fitted distribution

If the original run requested too few draws per run, we can generate fresh draws from the already-fitted mixture, avoiding re-running the optimization step, and optionally importance-resample them.
The `ndraws_per_run` keyword controls how many fresh draws to sample from each mixture component before resampling.

```@example 1
result2 = resample(result, 2_000; ndraws_per_run=200)
```

The Pareto diagnostic for this new result reflects the quality of importance resampling over the fresh draws.
The importance-weighting step requires evaluating the provided log-density on all draws, which can be very expensive if the log-density itself is expensive.
If you have strong evidence that Pathfinder's approximation is sufficiently close to your target distribution that you don't need importance reweighting or the Pareto ``k`` diagnostic, you may skip importance resampling entirely and draw directly from the fitted mixture distribution, pass `importance=false`:

```@example 1
result3 = resample(result, 2_000; ndraws_per_run=200, importance=false)
```
