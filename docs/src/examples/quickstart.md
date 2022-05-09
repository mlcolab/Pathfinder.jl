# Quickstart

This page introduces basic Pathfinder usage with examples.

## A 5-dimensional multivariate normal

For a simple example, we'll run Pathfinder on a multivariate normal distribution with
a dense covariance matrix.

```@example 1
using LinearAlgebra, Pathfinder, Printf, StatsPlots, Random
Random.seed!(42)

Σ = [
    2.71   0.5    0.19   0.07   1.04
    0.5    1.11  -0.08  -0.17  -0.08
    0.19  -0.08   0.26   0.07  -0.7
    0.07  -0.17   0.07   0.11  -0.21
    1.04  -0.08  -0.7   -0.21   8.65
]
μ = [-0.55, 0.49, -0.76, 0.25, 0.94]
P = inv(Symmetric(Σ))

function logp_mvnormal(x)
    z = x - μ
    return -dot(z, P, z) / 2
end
nothing # hide
```

Now we run [`pathfinder`](@ref).

```@example 1
result = pathfinder(logp_mvnormal; dim=5, init_scale=4)
```

`result` is a [`PathfinderResult`](@ref).
See its docstring for a description of its fields.

`result.fit_distribution` is a multivariate normal approximation to our target distribution.
Its mean and covariance are quite close to our target distribution's.

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

We can visualize Pathfinder's sequence of multivariate-normal approximations with the following function:

```@example 1
function plot_pathfinder_trace(
    result, logp_marginal, xrange, yrange, maxiters;
    show_elbo=false, flipxy=false, kwargs...,
)
    iterations = min(length(result.optim_trace) - 1, maxiters)
    trace_points = result.optim_trace.points
    trace_dists = result.fit_distributions
    anim = @animate for i in 1:iterations
        contour(xrange, yrange, exp ∘ logp_marginal ∘ Base.vect; label="")
        trace = trace_points[1:(i + 1)]
        dist = trace_dists[i + 1]
        plot!(
            first.(trace), last.(trace);
            seriestype=:scatterpath, color=:black, msw=0, label="trace",
        )
        covellipse!(
            dist.μ[1:2], dist.Σ[1:2, 1:2];
            n_std=2.45, alpha=0.7, color=1, linecolor=1, label="MvNormal 95% ellipsoid",
        )
        title = "Iteration $i"
        if show_elbo
            est = result.elbo_estimates[i]
            title *= "  ELBO estimate: " * @sprintf("%.1f", est.value)
        end
        plot!(; xlims=extrema(xrange), ylims=extrema(yrange), title, kwargs...)
    end
    return anim
end;
nothing #hide
```

```@example 1
xrange = -5:0.1:5
yrange = -5:0.1:5

μ_marginal = μ[1:2]
P_marginal = inv(Σ[1:2,1:2])
logp_mvnormal_marginal(x) = -dot(x - μ_marginal, P_marginal, x - μ_marginal) / 2

anim = plot_pathfinder_trace(
    result, logp_mvnormal_marginal, xrange, yrange, 20;
    xlabel="x₁", ylabel="x₂",
)
gif(anim; fps=5)
```

## A banana-shaped distribution

Now we will run Pathfinder on the following banana-shaped distribution:

```math
\pi(x_1, x_2) = e^{-x_1^2 / 2} e^{-5 (x_2 - x_1^2)^2 / 2}
```

First we define the distribution,

```@example 1
Random.seed!(23)

logp_banana(x) = -(x[1]^2 + 5(x[2] - x[1]^2)^2) / 2
nothing # hide
```

and then visualise it:

```@example 1
xrange = -3.5:0.05:3.5
yrange = -3:0.05:7
contour(xrange, yrange, exp ∘ logp_banana ∘ Base.vect; xlabel="x₁", ylabel="x₂")
```

Now we run [`pathfinder`](@ref).

```@example 1
result = pathfinder(logp_banana; dim=2, init_scale=10)
```

As before we can visualise each iteration of the algorithm.

```@example 1
anim = plot_pathfinder_trace(
    result, logp_banana, xrange, yrange, 20;
    xlabel="x₁", ylabel="x₂",
)
gif(anim; fps=5)
```

Especially for complicated target distributions, it's more useful to run multi-path Pathfinder.

We can see that most of the approximations above are not great, because this distribution is far from normal. It is always a good idea to run [`multipathfinder`](@ref) directly, which runs single-path Pathfinder multiple times.

```@example 1
ndraws = 1_000
result = multipathfinder(logp_banana, ndraws; nruns=20, dim=2, init_scale=10)
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

contour(xrange, yrange, (x, y) -> exp(logp_banana([x, y])))
scatter!(x₁_approx, x₂_approx; msw=0, ms=2, alpha=0.5, color=1)
plot!(xlims=extrema(xrange), ylims=extrema(yrange), xlabel="x₁", ylabel="x₂", legend=false)
```

## A 100-dimensional funnel

As we have seen above, running multi-path Pathfinder is much more useful for target distributions that are far from normal.
One particularly difficult distribution to sample is Neal's funnel:

```math
\begin{aligned}
\tau &\sim \mathrm{Normal}(\mu=0, \sigma=3)\\
\beta_i &\sim \mathrm{Normal}(\mu=0, \sigma=e^{\tau/2})
\end{aligned}
```

Such funnel geometries appear in other models (e.g. many hierarchical models) and typically frustrate MCMC sampling.
Multi-path Pathfinder can't sample the funnel well, but it can quickly give us draws that can help us diagnose that we have a funnel.

In this example, we draw from a 100-dimensional funnel and visualize 2 dimensions.

```@example 1
Random.seed!(68)

function logp_funnel(x)
    n = length(x)
    τ = x[1]
    β = view(x, 2:n)
    return ((τ / 3)^2 + (n - 1) * τ + sum(b -> abs2(b * exp(-τ / 2)), β)) / -2
end

dim = 100
init_scale = 10
nothing # hide
```

First, let's fit this posterior with single-path Pathfinder.

```@example 1
result_single = pathfinder(logp_funnel; dim, init_scale)
```

The L-BFGS optimizer constructs an approximation to the inverse Hessian of the negative log density using the limited history of previous points and gradients.
For each iteration, Pathfinder uses this estimate as an approximation to the covariance matrix of a multivariate normal that approximates the target distribution.
The distribution that maximizes the evidence lower bound (ELBO) is returned.

Let's visualize this sequence of multivariate normals for the first two dimensions.

```@example 1
β₁_range = -5:0.01:5
τ_range = -15:0.01:5

anim = plot_pathfinder_trace(
    result_single, logp_funnel, τ_range, β₁_range, 15;
    show_elbo=true, xlabel="τ", ylabel="β₁",
)
gif(anim; fps=2)
```

For this challenging posterior, we can again see that most of the approximations are not great, because this distribution is not normal.
Also, this distribution has a pole instead of a mode, so there is no MAP estimate, and no Laplace distribution exists.
As optimization proceeds, the approximation goes from very bad to less bad and finally extremely bad.
The ELBO-maximizing distribution is at the neck of the funnel, which would be a good location to initialize MCMC.

It is always a good idea to run [`multipathfinder`](@ref) directly, which runs single-path Pathfinder multiple times.

```@example 1
ndraws = 1_000
result = multipathfinder(logp_funnel, ndraws; nruns=20, dim, init_scale)
```

Again, PSIS.jl's warning indicates we should follow this up with MCMC.

Here we can see that the bulk of Pathfinder's draws come from the neck of the funnel, where the fit from the single path we examined was located.

```@example 1
τ_approx = result.draws[1, :]
β₁_approx = result.draws[2, :]

contour(τ_range, β₁_range, exp ∘ logp_funnel ∘ Base.vect)
scatter!(τ_approx, β₁_approx; msw=0, ms=2, alpha=0.5, color=1)
plot!(; xlims=extrema(τ_range), ylims=extrema(β₁_range), xlabel="τ", ylabel="β₁", legend=false)
```
