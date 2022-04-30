# Multi-path Pathfinder

```@docs
multipathfinder
```

## Examples

Especially for complicated target distributions, it's more useful to run multi-path Pathfinder.
One difficult distribution to sample is Neal's funnel:

```math
\tau \sim \mathrm{Normal}(0, 3)\\
\beta_i \sim \mathrm{Normal}(0, e^{\tau/2})
```

Such funnel geometries appear in other models and typically frustrate MCMC sampling.
Multi-path Pathfinder can't sample the funnel well, but it can quickly give us draws that can help us diagnose that we have a funnel.

In this example, we draw from a 100-dimensional funnel and visualize 2 dimensions.

```@example 1
using ForwardDiff, Pathfinder, Random

Random.seed!(42)

function logp(x)
    n = length(x)
    τ = x[1]
    β = view(x, 2:n)
    return ((τ / 3)^2 + (n - 1) * τ + sum(b -> abs2(b * exp(-τ / 2)), β)) / -2
end
∇logp(x) = ForwardDiff.gradient(logp, x)

dim = 100
ndraws = 1_000
nruns = 20
nothing # hide
```

Now we run multi-path Pathfinder.

```@repl 1
@time q, ϕ, component_ids = multipathfinder(logp, ∇logp, ndraws; nruns, dim, init_scale=10);
```

The first return value is a uniformly-weighted `Distributions.MixtureModel`.
Each component is the result of a single Pathfinder run.

```@repl 1
typeof(q)
```

While we could draw samples from `q` directly, these aren't equivalent to the samples returned by multi-path Pathfinder, which uses multiple importance sampling with Pareto-smoothed importance resampling to combine the individual runs.

Note that [PSIS.jl](https://github.com/arviz-devs/PSIS.jl), which smooths the importance weights, warns us that the importance weights are unsuitable for computing estimates, so we should definitely run MCMC to get better samples.
Pathfinder's samples can still be useful though for initializing MCMC.

Let's compare Pathfinder's samples with samples from `q` directly, plotted over the exact marginal density.

```@example 1
using Plots

τ_approx = ϕ[1, :]
β₁_approx = ϕ[2, :]
ϕ2 = rand(q, ndraws)
τ_approx2 = ϕ2[1, :]
β₁_approx2 = ϕ2[2, :]

τ_range = -15:0.01:5
β₁_range = -5:0.01:5

plt = contour(β₁_range, τ_range, (β, τ) -> exp(logp([τ, β])))
scatter!(β₁_approx2, τ_approx2; msw=0, ms=2, alpha=0.3, color=:blue, label="Pathfinder w/o IR")
scatter!(β₁_approx, τ_approx; msw=0, ms=2, alpha=0.5, color=:red, label="Pathfinder")
plot!(xlims=extrema(β₁_range), ylims=extrema(τ_range), xlabel="β₁", ylabel="τ", legend=true)
plt
```

As expected from PSIS.jl's warning, neither set of samples cover the density well.
But we can also see that the mixture model (`q`) places too many samples in regions of low density, which is corrected by the importance resampling.

We can check how much each component contributed to the returned sample.

```@example 1
histogram(component_ids; bins=(0:nruns) .+ 0.5, bar_width=0.8, xticks=1:nruns,
          xlabel="Component index", ylabel="Count", legend=false)
```
