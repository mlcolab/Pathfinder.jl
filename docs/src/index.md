```@meta
CurrentModule = Pathfinder
```

# Pathfinder.jl: Parallel quasi-Newton variational inference

This package implements Pathfinder, [^Zhang2021] a variational method for initializing Markov chain Monte Carlo (MCMC) methods.

## Single-path Pathfinder

Single-path Pathfinder ([`pathfinder`](@ref)) attempts to return draws in or near the typical set, usually with many fewer gradient evaluations.
Pathfinder uses the [limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)(L-BFGS) optimizer to construct a _maximum a posteriori_ (MAP) estimate of a target posterior distribution ``p``.
It then uses the trace of the optimization to construct a sequence of multivariate normal approximations to the target distribution, returning the approximation that maximizes the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) -- equivalently, minimizes the [Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) divergence from the target distribution -- as well as draws from the distribution.

## Multi-path Pathfinder

Multi-path Pathfinder ([`multipathfinder`](@ref)) consists of running Pathfinder multiple times.
It returns a uniformly-weighted mixture model of the multivariate normal approximations of the individual runs.
It also uses importance resampling to return samples that better approximate the target distribution and assess the quality of the approximation.

## Uses

### Using the Pathfinder draws

!!! note "Folk theorem of statistical computing"
    When you have computational problems, often there’s a problem with your model.

Visualizing posterior draws is a common way to diagnose problems with a model. 
However, problematic models often tend to be slow to warm-up.
Even if the draws returned by Pathfinder are only approximations to the posterior, they can sometimes still be used to diagnose basic issues such as highly correlated parameters, parameters with very different posterior variances, and multimodality.

### Initializing MCMC

Pathfinder can be used to initialize MCMC.
This especially useful when sampling with Hamiltonian Monte Carlo.
See [Initializing HMC with Pathfinder](@ref) for details.

## Integration with the Julia ecosystem

Pathfinder uses several packages for extended functionality:

- [GalacticOptim.jl](https://galacticoptim.sciml.ai/stable/): This allows the L-BFGS optimizer to be replaced with any of the many GalacticOptim-compatible optimizers and supports use of callbacks. Note that any changes made to Pathfinder using these features would be experimental.
- [Transducers.jl](https://juliafolds.github.io/Transducers.jl/stable/): parallelization support
- [Distributions.jl](https://juliastats.org/Distributions.jl/stable/)/[PDMats.jl](https://github.com/JuliaStats/PDMats.jl): fits can be used anywhere a `Distribution` can be used
- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl): selecting the AD package used to differentiate the provided log-density function.
- [ProgressLogging.jl](https://julialogging.github.io/ProgressLogging.jl/stable/): In Pluto, Juno, and VSCode, nested progress bars are shown. In the REPL, use TerminalLoggers.jl to get progress bars.

[^Zhang2021]: Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).
              Pathfinder: Parallel quasi-Newton variational inference.
              arXiv: [2108.03782](https://arxiv.org/abs/2108.03782) [stat.ML].
              [Code](https://github.com/LuZhangstat/Pathfinder)
