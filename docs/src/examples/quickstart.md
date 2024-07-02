# Quickstart

This page introduces basic Pathfinder usage with examples.

## A 5-dimensional multivariate normal

For a simple example, we'll run Pathfinder on a multivariate normal distribution with
a dense covariance matrix.
Pathfinder can take a log-density function.
By default, the gradient of the log-density function is computed using ForwardDiff.

```@example 1
using ADTypes, ForwardDiff, LinearAlgebra, LogDensityProblems,
      Pathfinder, Printf, ReverseDiff, StatsPlots, Random
Random.seed!(42)

struct MvNormalProblem{T,S}
    μ::T  # mean
    P::S  # precision matrix
end
function (prob::MvNormalProblem)(x)
    z = x - prob.μ
    return -dot(z, prob.P, z) / 2
end

Σ = [
    2.71   0.5    0.19   0.07   1.04
    0.5    1.11  -0.08  -0.17  -0.08
    0.19  -0.08   0.26   0.07  -0.7
    0.07  -0.17   0.07   0.11  -0.21
    1.04  -0.08  -0.7   -0.21   8.65
]
μ = [-0.55, 0.49, -0.76, 0.25, 0.94]
P = inv(Symmetric(Σ))
prob_mvnormal = MvNormalProblem(μ, P)

nothing # hide
```

Now we run [`pathfinder`](@ref).

```@example 1
result = pathfinder(prob_mvnormal; dim=5, init_scale=4)
```

`result` is a [`PathfinderResult`](@ref).
See its docstring for a description of its fields.

The L-BFGS optimizer constructs an approximation to the inverse Hessian of the negative log density using the limited history of previous points and gradients.
For each iteration, Pathfinder uses this estimate as an approximation to the covariance matrix of a multivariate normal that approximates the target distribution.
The distribution that maximizes the evidence lower bound (ELBO) is stored in `result.fit_distribution`.
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
            first.(trace), getindex.(trace, 2);
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

Now we will run Pathfinder on the following banana-shaped distribution with density

```math
\pi(x_1, x_2) = e^{-x_1^2 / 2} e^{-5 (x_2 - x_1^2)^2 / 2}.
```

Pathfinder can also take any object that implements the [LogDensityProblems](https://www.tamaspapp.eu/LogDensityProblems.jl) interface.
This can also be used to manually define the gradient of the log-density function.

First we define the log density problem:

```@example 1
Random.seed!(23)

struct BananaProblem end
function LogDensityProblems.capabilities(::Type{<:BananaProblem})
    return LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.dimension(::BananaProblem) = 2
function LogDensityProblems.logdensity(::BananaProblem, x)
    return -(x[1]^2 + 5(x[2] - x[1]^2)^2) / 2
end
function LogDensityProblems.logdensity_and_gradient(::BananaProblem, x)
    a = (x[2] - x[1]^2)
    lp = -(x[1]^2 + 5a^2) / 2
    grad_lp = [(10a - 1) * x[1], -5a]
    return lp, grad_lp
end

prob_banana = BananaProblem()

nothing # hide
```

and then visualise it:

```@example 1
xrange = -3.5:0.05:3.5
yrange = -3:0.05:7
logp_banana(x) = LogDensityProblems.logdensity(prob_banana, x)
contour(xrange, yrange, exp ∘ logp_banana ∘ Base.vect; xlabel="x₁", ylabel="x₂")
```

Now we run [`pathfinder`](@ref).

```@example 1
result = pathfinder(prob_banana; init_scale=10)
```

As before we can visualise each iteration of the algorithm.

```@example 1
anim = plot_pathfinder_trace(
    result, logp_banana, xrange, yrange, 20;
    xlabel="x₁", ylabel="x₂",
)
gif(anim; fps=5)
```

Since the distribution is far from normal, Pathfinder is unable to fit the distribution well.
Especially for such complicated target distributions, it's always a good idea to run [`multipathfinder`](@ref), which runs single-path Pathfinder multiple times.

```@example 1
ndraws = 1_000
result = multipathfinder(prob_banana, ndraws; nruns=20, init_scale=10)
```

`result` is a [`MultiPathfinderResult`](@ref).
See its docstring for a description of its fields.

`result.fit_distribution` is a uniformly-weighted `Distributions.MixtureModel`.
Each component is the result of a single Pathfinder run.
It's possible that some runs fit the target distribution much better than others, so instead of just drawing samples from `result.fit_distribution`, `multipathfinder` draws many samples from each component and then uses Pareto-smoothed importance resampling (using [PSIS.jl](https://psis.julia.arviz.org/stable/)) from these draws to better target `prob_banana`.

The Pareto shape diagnostic informs us on the quality of these draws.
Here the Pareto shape ``k`` diagnostic is bad (``k > 0.7``), which tells us that these draws are unsuitable for computing posterior estimates, so we should definitely run MCMC to get better draws.
Still, visualizing the draws can still be useful.

```@example 1
x₁_approx = result.draws[1, :]
x₂_approx = result.draws[2, :]

contour(xrange, yrange, exp ∘ logp_banana ∘ Base.vect)
scatter!(x₁_approx, x₂_approx; msw=0, ms=2, alpha=0.5, color=1)
plot!(xlims=extrema(xrange), ylims=extrema(yrange), xlabel="x₁", ylabel="x₂", legend=false)
```

While the draws do a poor job of covering the tails of the distribution, they are still useful for identifying the nonlinear correlation between these two parameters.

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
using ReverseDiff, ADTypes

Random.seed!(68)

function logp_funnel(x)
    n = length(x)
    τ = x[1]
    β = view(x, 2:n)
    return ((τ / 3)^2 + (n - 1) * τ + sum(b -> abs2(b * exp(-τ / 2)), β)) / -2
end

nothing # hide
```

First, let's fit this posterior with single-path Pathfinder.
For high-dimensional problems, it's better to use reverse-mode automatic differentiation.
Here, we'll use `ADTypes.AutoReverseDiff()` to specify that [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) should be used.


```@example 1
result_single = pathfinder(logp_funnel; dim=100, init_scale=10, adtype=AutoReverseDiff())
```

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
Also, this distribution has a pole instead of a mode, so there is no MAP estimate, and no Laplace approximation exists.
As optimization proceeds, the approximation goes from very bad to less bad and finally extremely bad.
The ELBO-maximizing distribution is at the neck of the funnel, which would be a good location to initialize MCMC.

Now we run [`multipathfinder`](@ref).

```@example 1
ndraws = 1_000
result = multipathfinder(logp_funnel, ndraws; dim=100, nruns=20, init_scale=10, adtype=AutoReverseDiff())
```

Again, the poor Pareto shape diagnostic indicates we should run MCMC to get draws suitable for computing posterior estimates.

We can see that the bulk of Pathfinder's draws come from the neck of the funnel, where the fit from the single path we examined was located.

```@example 1
τ_approx = result.draws[1, :]
β₁_approx = result.draws[2, :]

contour(τ_range, β₁_range, exp ∘ logp_funnel ∘ Base.vect)
scatter!(τ_approx, β₁_approx; msw=0, ms=2, alpha=0.5, color=1)
plot!(; xlims=extrema(τ_range), ylims=extrema(β₁_range), xlabel="τ", ylabel="β₁", legend=false)
```
