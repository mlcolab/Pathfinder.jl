# Running Pathfinder on Turing.jl models

This tutorial demonstrates how [Turing](https://turinglang.org/) can be used with Pathfinder.

We'll demonstrate with a regression example.

```@example 1
using AdvancedHMC, Pathfinder, Random, Turing
Random.seed!(39)

@model function regress(x)
    α ~ Normal()
    β ~ Normal()
    σ ~ truncated(Normal(); lower=0)
    μ = α .+ β .* x
    y ~ product_distribution(Normal.(μ, σ))
end
x = 0:0.1:10
true_params = (; α=1.5, β=2, σ=2)
# simulate data
(; y) = rand(regress(x) | true_params)

model = regress(x) | (; y)
n_chains = 8
nothing # hide
```

For convenience, [`pathfinder`](@ref) and [`multipathfinder`](@ref) can take [Turing models](@extref `DynamicPPL.Model`) as inputs and produce [`MCMCChains.Chains`](@extref) objects as outputs.
To access this, we run Pathfinder normally; the `Chains` representation of the draws is stored in `draws_transformed`.

```@example 1
result_single = pathfinder(model; ndraws=1_000)
```

```@example 1
result_single.draws_transformed
```

Note that while Turing's `sample` methods default to initializing parameters from the prior with [`InitFromPrior`](@extref `DynamicPPL.InitFromPrior`), Pathfinder defaults to uniformly sampling them in the range `[-2, 2]` in unconstrained space (equivalent to Turing's [`InitFromUniform(-2, 2)`](@extref `DynamicPPL.InitFromUniform`)).
To use Turing's default in Pathfinder, specify `init_sampler=InitFromPrior()`.

```@example 1
result_multi = multipathfinder(model, 1_000; nruns=n_chains, init_sampler=InitFromPrior())
```

The Pareto shape diagnostic indicates that it is likely safe to use these draws to compute posterior estimates.

```@example 1
chns_pf = result_multi.draws_transformed
describe(chns_pf)
```

We can also use these draws to initialize MCMC sampling with [`InitFromParams`](@extref `DynamicPPL.InitFromParams`).

```@example 1
var_names = names(chns_pf, :parameters)
initial_params = [
    InitFromParams(get(chns_pf[i, :, :], var_names; flatten=true)) for i in 1:n_chains
]
nothing # hide
```

```@example 1
chns = sample(model, Turing.NUTS(), MCMCThreads(), 1_000, n_chains; initial_params, progress=false)
describe(chns)
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
    initial_params,
    progress=false,
)[n_adapts + 1:end, :, :]  # drop warm-up draws
describe(chns)
```

See [Initializing HMC with Pathfinder](@ref) for further examples.
