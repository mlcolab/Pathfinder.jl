# Running Pathfinder on Turing.jl models

This tutorial demonstrates how [Turing](https://turinglang.org/) can be used with Pathfinder.

We'll demonstrate with a regression example.

```@example 1
using AbstractMCMC, AdvancedHMC, DynamicPPL, FlexiChains, MCMCChains, Pathfinder, Random, Turing
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
y = rand(regress(x) | true_params)[@varname(y)]

model = regress(x) | (; y)
n_chains = 8
nothing # hide
```

For convenience, [`pathfinder`](@ref) and [`multipathfinder`](@ref) can take [Turing models](@extref `DynamicPPL.Model`) as inputs and produce [`FlexiChains.VNChain`](@extref `FlexiChains.FlexiChain`) objects as outputs.
To access this, we run Pathfinder normally; the chains representation of the draws is stored in `draws_transformed`, with the type defaulting to whatever `Turing.sample` itself defaults to.

!!! note "Turing pre-v0.45 backward-compatibility"
    Since Turing v0.45, the default `chain_type` is [`FlexiChains.VNChain`](@extref `FlexiChains.FlexiChain`), while previous versions returned [`MCMCChains.Chains`](@extref).
    Pathfinder will return the same default chain type that your installed Turing version returns, but you can always specify the `chain_type` manually.
    This tutorial uses the new default (FlexiChains) throughout; see [Using MCMCChains](@ref) below if you still want `MCMCChains.Chains`.

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
summarystats(chns_pf)
```

We can also use these draws to initialize MCMC sampling with [`InitFromParams`](@extref `DynamicPPL.InitFromParams`).
[`FlexiChains.VNChain`](@extref `FlexiChains.FlexiChain`) subsets iterations and chains with keyword arguments rather than [`MCMCChains.Chains`](@extref)'s 3-argument indexing; see the [FlexiChains migration guide](https://pysm.dev/FlexiChains.jl/stable/migration/) for more such translations.

```@example 1
params = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, chns_pf[iter=1:n_chains], model)
initial_params = [InitFromParams(p.params) for p in vec(params)]
nothing # hide
```

```@example 1
chns = sample(model, Turing.NUTS(), MCMCThreads(), 1_000, n_chains; initial_params, progress=false)
summarystats(chns)
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
)[iter=(n_adapts + 1):(n_adapts + n_draws)]  # drop warm-up draws
summarystats(chns)
```

See [Initializing HMC with Pathfinder](@ref) for further examples.

## Using MCMCChains

To request transformed draws be returned as [`MCMCChains.Chains`](@extref) instead of the default, you may specify `chain_type` directly.

```@example 1
result_multi_mcmc = multipathfinder(
    model, 1_000; nruns=n_chains, init_sampler=InitFromPrior(), chain_type=MCMCChains.Chains
)
chns_pf_mcmc = result_multi_mcmc.draws_transformed
```

As before, we can use these draws to initialize MCMC sampling with [`InitFromParams`](@extref `DynamicPPL.InitFromParams`).
Note that `Chains` subsets iterations and chains with 3-argument indexing (`chain[iterations, parameters, chains]`): 

```@example 1
params_mcmcchains = AbstractMCMC.to_samples(
    DynamicPPL.ParamsWithStats, chns_pf_mcmc[1:n_chains, :, :], model
)
initial_params_mcmcchains = [InitFromParams(p.params) for p in vec(params_mcmcchains)]
nothing # hide
```
