# Running Pathfinder on Turing.jl models

This tutorial demonstrates how Turing can be used with Pathfinder.

We'll demonstrate with a regression example.

```@example 1
using AdvancedHMC, LinearAlgebra, Pathfinder, Random, Turing
Random.seed!(39)

@model function regress(x, y)
    α ~ Normal()
    β ~ Normal()
    σ ~ truncated(Normal(); lower=0)
    y .~ Normal.(α .+ β .* x, σ)
end
x = 0:0.1:10
y = @. 2x + 1.5 + randn.() * 0.2
nothing # hide
```

```@example 1
model = regress(collect(x), y)
```

The first way we can use Turing with Pathfinder is via its mode estimation functionality.
We can use `Turing.optim_problem` to generate a `SciMLBase.OptimizationFunction`, which [`pathfinder`](@ref) and [`multipathfinder`](@ref) can take as inputs.

```@example 1
fun = optim_function(model, MAP(); constrained=false)
```

```@example 1
dim = length(fun.init())
pathfinder(fun.func; dim)
```

```@example 1
multipathfinder(fun.func, 1_000; dim, nruns=8)
```

However, for convenience, `pathfinder` and `multipathfinder` can take Turing models as inputs and produce `MCMCChains.Chains` objects as outputs.

```@example 1
result_single = pathfinder(model; ndraws=1_000)
```

```@example 1
result_multi = multipathfinder(model, 1_000; nruns=8)
```

Here, the Pareto shape diagnostic indicates that it is likely safe to use these draws to compute posterior estimates.

When passed a `Model`, Pathfinder also gives access to the posterior draws in a familiar `MCMC.Chains` object.

```@example 1
result_multi.draws_transformed
```

We can also use these posterior draws to initialize MCMC sampling.

```@example 1
init_params = collect.(eachrow(result_multi.draws_transformed.value[1:4, :, 1]))
```

```@example 1
chns = sample(model, Turing.NUTS(), MCMCThreads(), 1_000, 4; init_params, progress=false)
```

To use Pathfinder's estimate of the metric and skip warm-up, at the moment one needs to use AdvancedHMC directly.

```@example 1
ℓπ(x) = -fun.func.f(x, nothing)
function ∂ℓπ∂θ(x)
    g = similar(x)
    fun.func.grad(g, x, nothing)
    rmul!(g, -1)
    return ℓπ(x), g
end

ndraws = 1_000
nadapts = 50
inv_metric = result_multi.pathfinder_results[1].fit_distribution.Σ
metric = Pathfinder.RankUpdateEuclideanMetric(inv_metric)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
ϵ = find_good_stepsize(hamiltonian, init_params[1])
integrator = Leapfrog(ϵ)
proposal = AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
adaptor = StepSizeAdaptor(0.8, integrator)
samples, stats = sample(
    hamiltonian,
    proposal,
    init_params[1],
    ndraws,
    adaptor,
    nadapts;
    drop_warmup=true,
    progress=false,
)
```

See [Initializing HMC with Pathfinder](@ref) for further examples.
