# Single-path Pathfinder

```@docs
pathfinder
```

## Examples

For a simple example, we'll run Pathfinder on a multivariate normal distribution with
a dense covariance matrix.

```@repl
using LinearAlgebra, Pathfinder, Random
Random.seed!(42)
Σ = [
    2.71   0.5    0.19   0.07   1.04
    0.5    1.11  -0.08  -0.17  -0.08
    0.19  -0.08   0.26   0.07  -0.7
    0.07  -0.17   0.07   0.11  -0.21
    1.04  -0.08  -0.7   -0.21   8.65
];
μ = [-0.55, 0.49, -0.76, 0.25, 0.94];
P = inv(Symmetric(Σ));
function logp(x)
    z = x - μ
    return -dot(z, P, z) / 2;
end;
∇logp(x) = P * (μ - x);
θ₀ = rand(5) .* 4 .- 2  # θ₀ ~ Uniform(-2, 2)
pathfinder(logp, ∇logp, θ₀, 100; ndraws_elbo=100); # hide
@time q, ϕ, logqϕ = pathfinder(logp, ∇logp, θ₀, 100; ndraws_elbo=100);
q
q.μ
q.Σ
```
