# Initializing HMC with Pathfinder

## The MCMC warm-up phase

When using MCMC to draw samples from some target distribution, there is often a lengthy warm-up phase with 2 phases:
1. converge to the _typical set_ (the region of the target distribution where the bulk of the probability mass is located)
2. adapt any tunable parameters of the MCMC sampler (optional)

While (1) often happens fairly quickly, (2) usually requires a lengthy exploration of the typical set to iteratively adapt parameters suitable for further exploration.
An example is the widely used windowed adaptation scheme of Hamiltonian Monte Carlo (HMC) in Stan [StanHMCParameters](@citep), where a step size and positive definite metric (aka mass matrix) are adapted.
For posteriors with complex geometry, the adaptation phase can require many evaluations of the gradient of the log density function of the target distribution.

Pathfinder can be used to initialize MCMC, and in particular HMC, in 3 ways:
1. Initialize MCMC from one of Pathfinder's draws (replace phase 1 of the warm-up).
2. Initialize an HMC metric adaptation from the inverse of the covariance of the multivariate normal approximation (replace the first window of phase 2 of the warm-up).
3. Use the inverse of the covariance as the metric without further adaptation (replace phase 2 of the warm-up).

This tutorial demonstrates all three approaches with [DynamicHMC.jl](https://tamaspapp.eu/DynamicHMC.jl/stable/) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl).
Both of these packages have standalone implementations of adaptive HMC (aka NUTS) and can be used independently of any probabilistic programming language (PPL).
Both the [Turing](https://turinglang.org/) and [Soss](https://github.com/cscherrer/Soss.jl) PPLs have some DynamicHMC integration, while Turing also integrates with AdvancedHMC.

For demonstration purposes, we'll use the following dummy data:

```@example 1
using LinearAlgebra, Pathfinder, Random, StatsFuns, StatsPlots

Random.seed!(91)

x = 0:0.01:1
y = @. sin(10x) + randn() * 0.2 + x

scatter(x, y; xlabel="x", ylabel="y", legend=false, msw=0, ms=2)
```

We'll fit this using a simple polynomial regression model:

```math
\begin{aligned}
\sigma &\sim \text{Half-Normal}(\mu=0, \sigma=1)\\
\alpha, \beta_j &\sim \mathrm{Normal}(\mu=0, \sigma=1)\\
\hat{y}_i &= \alpha + \sum_{j=1}^J x_i^j \beta_j\\
y_i &\sim \mathrm{Normal}(\mu=\hat{y}_i, \sigma=\sigma)
\end{aligned}
```

We just need to implement the log-density function of the resulting posterior.

```@example 1
struct RegressionProblem{X,Z,Y}
    x::X
    J::Int
    z::Z
    y::Y
end
function RegressionProblem(x, J, y)
    z = x .* (1:J)'
    return RegressionProblem(x, J, z, y)
end

function (prob::RegressionProblem)(θ)
    (; σ, α, β) = θ
    (; z, y) = prob
    lp = normlogpdf(σ) + logtwo
    lp += normlogpdf(α)
    lp += sum(normlogpdf, β)
    y_hat = muladd(z, β, α)
    lp += sum(eachindex(y_hat, y)) do i
        return normlogpdf(y_hat[i], σ, y[i])
    end
    return lp
end

J = 3
dim = J + 2
model = RegressionProblem(x, J, y)
ndraws = 1_000;
nothing # hide
```

## DynamicHMC.jl

To use DynamicHMC, we first need to transform our model to an unconstrained space using [TransformVariables.jl](https://tamaspapp.eu/TransformVariables.jl/stable/) and wrap it in a type that implements the [LogDensityProblems interface](@extref LogDensityProblems log-density-api) (see [DynamicHMC's worked example](@extref DynamicHMC worked_example)):

```@example 1
using DynamicHMC, ForwardDiff, LogDensityProblems, LogDensityProblemsAD, TransformVariables
using TransformedLogDensities: TransformedLogDensity

transform = as((σ=asℝ₊, α=asℝ, β=as(Array, J)))
P = TransformedLogDensity(transform, model)
∇P = ADgradient(:ForwardDiff, P)
```

Pathfinder can take any object that implements this interface.

```@example 1
result_pf = pathfinder(∇P)
```

```@example 1
init_params = result_pf.draws[:, 1]
```

```@example 1
inv_metric = result_pf.fit_distribution.Σ
```

### Initializing from Pathfinder's draws

Here we just need to pass one of the draws as the initial point `q`:

```@example 1
result_dhmc1 = mcmc_with_warmup(
    Random.default_rng(),
    ∇P,
    ndraws;
    initialization=(; q=init_params),
    reporter=NoProgressReport(),
)
```

### Initializing metric adaptation from Pathfinder's estimate

To start with Pathfinder's inverse metric estimate, we just need to initialize a [`DynamicHMC.GaussianKineticEnergy`](@extref) object with it as input: 

```@example 1
result_dhmc2 = mcmc_with_warmup(
    Random.default_rng(),
    ∇P,
    ndraws;
    initialization=(; q=init_params, κ=GaussianKineticEnergy(inv_metric)),
    warmup_stages=default_warmup_stages(; M=Symmetric),
    reporter=NoProgressReport(),
)
```

We also specified that DynamicHMC should tune a dense `Symmetric` matrix.
However, we could also have requested a `Diagonal` metric.

### Use Pathfinder's metric estimate for sampling

To turn off metric adaptation entirely and use Pathfinder's estimate, we just set the number and size of the metric adaptation windows to 0.

```@example 1
result_dhmc3 = mcmc_with_warmup(
    Random.default_rng(),
    ∇P,
    ndraws;
    initialization=(; q=init_params, κ=GaussianKineticEnergy(inv_metric)),
    warmup_stages=default_warmup_stages(; middle_steps=0, doubling_stages=0),
    reporter=NoProgressReport(),
)
```

## AdvancedHMC.jl

Similar to Pathfinder and DynamicHMC, AdvancedHMC can also work with a LogDensityProblem:

```@example 1
using AdvancedHMC

nadapts = 500;
nothing # hide
```

### Initializing from Pathfinder's draws

```@example 1
metric = DiagEuclideanMetric(dim)
hamiltonian = Hamiltonian(metric, ∇P)
ϵ = find_good_stepsize(hamiltonian, init_params)
integrator = Leapfrog(ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.8, integrator)
samples_ahmc1, stats_ahmc1 = sample(
    hamiltonian,
    kernel,
    init_params,
    ndraws + nadapts,
    adaptor,
    nadapts;
    drop_warmup=true,
    progress=false,
)
```

### Initializing metric adaptation from Pathfinder's estimate

We can't start with Pathfinder's inverse metric estimate directly.
Instead we need to first extract its diagonal for a `DiagonalEuclideanMetric` or make it dense for a `DenseEuclideanMetric`:

```@example 1
metric = DenseEuclideanMetric(Matrix(inv_metric))
hamiltonian = Hamiltonian(metric, ∇P)
ϵ = find_good_stepsize(hamiltonian, init_params)
integrator = Leapfrog(ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.8, integrator)
samples_ahmc2, stats_ahmc2 = sample(
    hamiltonian,
    kernel,
    init_params,
    ndraws + nadapts,
    adaptor,
    nadapts;
    drop_warmup=true,
    progress=false,
)
```

### Use Pathfinder's metric estimate for sampling

To use Pathfinder's metric with no metric adaptation, we need to use Pathfinder's own [`Pathfinder.RankUpdateEuclideanMetric`](@ref) type, which just wraps our inverse metric estimate for use with AdvancedHMC:

```@example 1
nadapts = 75
metric = Pathfinder.RankUpdateEuclideanMetric(inv_metric)
hamiltonian = Hamiltonian(metric, ∇P)
ϵ = find_good_stepsize(hamiltonian, init_params)
integrator = Leapfrog(ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.8, integrator)
samples_ahmc3, stats_ahmc3 = sample(
    hamiltonian,
    kernel,
    init_params,
    ndraws + nadapts,
    adaptor,
    nadapts;
    drop_warmup=true,
    progress=false,
)
```
