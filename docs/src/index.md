```@meta
CurrentModule = Pathfinder
```

# Pathfinder

Pathfinder[^Zhang2021] is a variational method for initializing Markov chain Monte Carlo (MCMC) methods.

## Introduction

When using MCMC to draw samples from some target distribution, there is often a length warm-up phase with 2 goals:
1. converge to the _typical set_ (the region of the target distribution where the bulk of the probability mass is located)
2. adapt any tunable parameters of the MCMC sampler (optional)

Typically (2) requires a lengthy exploration of the typical set.
An example is the widely used windowed adaptation scheme of Hamiltonian Monte Carlo (HMC), where a step size and mass matrix are adapted
For posteriors with complex geometry, the adaptation phase can require many evaluations of the gradient of the log density function of the target distribution.

Pathfinder attempts to return samples in or near the typical set with many fewer gradient evaluations.
Pathfinder uses [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) to construct a _maximum a posteriori_ (MAP) estimate of a target distribution ``p``.
It then uses the trace of the optimization to construct a sequence of multivariate normal approximations to the target distribution, returning the approximation that maximizes the evidence lower bound (ELBO), as well as draws from the distribution.
The covariance of the multivariate normal approximation can be used to instantiate the mass matrix adaptation in HMC.

Its extension, multi-path Pathfinder, runs Pathfinder multiple times.
It returns a uniformly-weighted mixture model of the multivariate normal approximations of the individual runs.
It also uses importance resampling to return samples that better approximate the target distribution.

See [Usage](@ref).

[^Zhang2021]: Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).
              Pathfinder: Parallel quasi-Newton variational inference.
              arXiv: [2108.03782](https://arxiv.org/abs/2108.03782) [stat.ML].
              [Code](https://github.com/LuZhangstat/Pathfinder)
