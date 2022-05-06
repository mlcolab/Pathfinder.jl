# Banana shaped distribution

This page shows basic Pathfinder usage on a banana shaped distribution.

We'll run Pathfinder on the following distribution:

```math
\begin{equation}
\pi(x_1,x_2) = e^{-x_1^2/4}e^{-(x_2-2(x_1^2-5))^2/2}
\end{equation}
```

First we will define the distribution,

```@example 1
using LinearAlgebra, Pathfinder, Printf, StatsPlots, Random
Random.seed!(42)

logp_banana(x) = -(x[1]^2 + (x[2] - 2*(x[1]^2 - 5))^2) / 2
nothing # hide
```

and then visualise it:

```@example 1
xrange = -3.5:0.05:3.5
yrange = -15:0.05:12
contour(xrange, yrange, (x, y) -> banana([x, y]), xlabel="x₁", ylabel="x₂")
```

Now we run [`pathfinder`](@ref).

```@example 1
result = pathfinder(logp_banana; dim=2, init_scale=4)
```

`result` is a [`PathfinderResult`](@ref).
See its docstring for a description of its fields.

`result.fit_distribution` is a multivariate normal approximation to our target distribution.

```@example 1
result.fit_distribution.μ
```

```@example 1
result.fit_distribution.Σ
```

`result.draws` is a `Matrix` whose columns are the requested draws from `result.fit_distribution`:
```@example 1
result.draws
```

```@example 1
iterations = length(result.optim_trace) - 1
trace_points = result.optim_trace.points
trace_dists = result.fit_distributions

xrange = -5:0.1:5
yrange = -35:0.1:25

anim = @animate for i in 1:iterations
    contour(xrange, yrange, (x, y) -> banana([x, y]), label="") # cutoff logp at -500 for better visualistation
    trace = trace_points[1:(i + 1)]
    dist = trace_dists[i + 1]
    plot!(first.(trace), last.(trace); seriestype=:scatterpath, color=:black, msw=0, label="trace")
    covellipse!(dist.μ, dist.Σ; n_std=2.45, alpha=0.7, color=1, linecolor=1, label="MvNormal 95% ellipsoid")
    title = "Iteration $i"
    plot!(; xlims=extrema(xrange), ylims=extrema(yrange), xlabel="x₁", ylabel="x₂", legend=:bottomright, title)
end
gif(anim, fps=5)
```

Especially for complicated target distributions, it's more useful to run multi-path Pathfinder.

Like in the [funnel example](quickstart#a-100-dimensional-funnel), we can see that most of the normal approximations above are not great, because this distribution is far from normal. It is always a good idea to run [`multipathfinder`](@ref) directly, which runs single-path Pathfinder multiple times.

```@example 1
ndraws = 1_000
result = multipathfinder(logp_banana, ndraws; nruns=20, dim=2, init_scale=4)
```

`result` is a [`MultiPathfinderResult`](@ref).
See its docstring for a description of its fields.

`result.fit_distribution` is a uniformly-weighted `Distributions.MixtureModel`.
Each component is the result of a single Pathfinder run.
It's possible that some runs fit the target distribution much better than others, so instead of just drawing samples from `result.fit_distribution`, `multipathfinder` draws many samples from each component and then uses Pareto-smoothed importance resampling from these draws to better target `logp_banana`.

The Pareto shape diagnostic also informs us on the quality of these draws.
Here [PSIS.jl](https://psis.julia.arviz.org/stable/), which smooths the importance weights, warns us that the importance weights are unsuitable for computing estimates, so we should definitely run MCMC to get better draws.


```@example 1
x₁_approx = result.draws[1, :]
x₂_approx = result.draws[2, :]

contour(xrange, yrange, (x, y) -> banana([x, y]))
scatter!(x₁_approx, x₂_approx; msw=0, ms=2, alpha=0.5, color=1)
plot!(xlims=extrema(xrange), ylims=extrema(yrange), xlabel="x₁", ylabel="x₂", legend=false)
```