# Running Pathfinder on Turing.jl models

This tutorial demonstrates how [Turing](https://turing.ml/stable/) can be used with Pathfinder.

We'll demonstrate with a regression example.

```@example 1
using AdvancedHMC, Pathfinder, Random, Turing
Random.seed!(39)

@model function regress(x, y)
    α ~ Normal()
    β ~ Normal()
    σ ~ truncated(Normal(); lower=0)
    y .~ Normal.(α .+ β .* x, σ)
end
x = 0:0.1:10
y = @. 2x + 1.5 + randn() * 0.2
nothing # hide
```

```@example 1
model = regress(collect(x), y)
n_chains = 8
```

For convenience, [`pathfinder`](@ref) and [`multipathfinder`](@ref) can take Turing models as inputs and produce `MCMCChains.Chains` objects as outputs.

```@example 1
result_single = pathfinder(model; ndraws=1_000)
```

```@example 1
result_multi = multipathfinder(model, 1_000; nruns=n_chains)
```

Here, the Pareto shape diagnostic indicates that it is likely safe to use these draws to compute posterior estimates.

When passed a `Model`, Pathfinder also gives access to the posterior draws in a familiar `MCMCChains.Chains` object.

```@example 1
result_multi.draws_transformed
```

We can also use these posterior draws to initialize MCMC sampling.

```@example 1
init_params = collect.(eachrow(result_multi.draws_transformed.value[1:n_chains, :, 1]))
```

```@example 1
chns = sample(model, Turing.NUTS(), MCMCThreads(), 1_000, n_chains; init_params, progress=false)
```

We can use Pathfinder's estimate of the metric and only perform enough warm-up to tune the step size.

```@example 1
inv_metric = result_multi.pathfinder_results[1].fit_distribution.Σ
metric = Pathfinder.RankUpdateEuclideanMetric(inv_metric)
kernel = HMCKernel(Trajectory{MultinomialTS}(Leapfrog(0.0), GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.8, 1.0)  # adapt only the step size
nuts = AdvancedHMC.HMCSampler(kernel, metric, adaptor)

n_adapts = 50
n_draws = 1_000
chns = sample(
    model,
    externalsampler(nuts),
    MCMCThreads(),
    n_draws + n_adapts,
    n_chains;
    n_adapts,
    init_params,
    progress=false,
)[n_adapts + 1:end, :, :]  # drop warm-up draws
```

See [Initializing HMC with Pathfinder](@ref) for further examples.
