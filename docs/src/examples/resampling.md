# Resampling from a fitted approximation

After running [`multipathfinder`](@ref), the fitted mixture distribution and individual-run draws
are stored in the result.
[`resample`](@ref) lets you draw new samples from this approximation without re-running Pathfinder.
Two common workflows are:

1. **Selecting unique MCMC initialization points**: When the Pareto shape ``\hat{k}`` is large,
   importance resampling *without replacement* yields at most one draw per unique candidate —
   suitable as distinct starting points for independent MCMC chains.
2. **Requesting more draws**: When the original run used too few `ndraws_per_run`, fresh draws
   can be generated from the fitted mixture and importance-resampled, without re-fitting.

## Setup

We use the banana-shaped distribution from the [Quickstart](@ref) as our running example,
since it has a poor Pareto-``\hat{k}`` diagnostic.

```@example 1
using Pathfinder, Random

Random.seed!(99)

logp_banana(x) = -(x[1]^2 + 5(x[2] - x[1]^2)^2) / 2

result = multipathfinder(logp_banana, 200; dim=2, nruns=20, init_scale=10)
```

The large Pareto shape ``\hat{k}`` indicates that these draws are unreliable for computing
posterior estimates, and we should run MCMC to get better samples.

```@example 1
result.psis_result.pareto_shape
```

## Workflow 1: unique MCMC initialization points

To seed ``N`` independent MCMC chains, we need ``N`` distinct starting points.
`resample` with `replace=false` selects unique draws — at most one per candidate — using the
stored PSIS weights, with no additional log-density evaluations.

```@example 1
nchains = 4
init_result = resample(result, nchains; replace=false)
```

Each column of `init_result.draws` is a distinct point from the original candidate pool:

```@example 1
init_result.draws
```

The resampled result preserves the fitted distribution and all other fields, so it can be
passed directly to any downstream Pathfinder-aware workflow:

```@example 1
init_result.fit_distribution === result.fit_distribution
```

## Workflow 2: more draws from the fitted distribution

If the original run requested too few draws per run, we can generate fresh draws from the
already-fitted mixture — avoiding a full Pathfinder re-run — and importance-resample them.
The `ndraws_per_run` keyword controls how many fresh draws to sample from each mixture component
before resampling.

```@example 1
result2 = resample(result, 2_000; ndraws_per_run=200)
size(result2.draws)
```

The Pareto diagnostic for this new result reflects the quality of importance resampling over
the fresh draws:

```@example 1
result2.psis_result.pareto_shape
```

To skip importance resampling entirely and draw directly from the mixture, pass
`importance=false`:

```@example 1
result3 = resample(result, 2_000; ndraws_per_run=200, importance=false)
result3.psis_result === nothing
```
