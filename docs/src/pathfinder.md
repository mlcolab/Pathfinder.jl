# Single-path Pathfinder

```@docs
pathfinder
PathfinderResult
```

## Examples

For a simple example, we'll run Pathfinder on a multivariate normal distribution with
a dense covariance matrix.

```@example 1
using LinearAlgebra, Pathfinder, Random
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

function logp(x)
    z = x - μ
    return -dot(z, P, z) / 2;
end
∇logp(x) = P * (μ - x)
nothing # hide
```

Now we run Pathfinder.

```@repl 1
result = pathfinder(logp, ∇logp; dim=5, ndraws=100, ndraws_elbo=100)
```

`result` is a [`PathfinderResult`](@ref).
See its docstring for a description of its fields.

`result.fit_dist_opt` is a multivariate normal approximation to our target distribution.
Its mean and covariance are quite close to our target distribution's.

```@repl 1
result.fit_dist_opt
result.fit_dist_opt.μ
result.fit_dist_opt.Σ
```

`result.draws` is a `Matrix` whose columns are the requested draws from `result.fit_dist_opt`:
```@repl 1
result.draws
```
