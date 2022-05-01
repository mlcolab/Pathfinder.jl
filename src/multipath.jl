"""
    multipathfinder(logp, ndraws; kwargs...)
    multipathfinder(logp, ∇logp, ndraws; kwargs...)
    multipathfinder(fun::GalacticOptim.OptimizationFunction, ndraws; kwargs...)

Run [`pathfinder`](@ref) multiple times to fit a multivariate normal mixture model.

For `nruns=length(init)`, `nruns` parallel runs of pathfinder produce `nruns` multivariate
normal approximations ``q_k = q(\\phi | \\mu_k, \\Sigma_k)`` of the posterior. These are
combined to a mixture model ``q`` with uniform weights.

``q`` is augmented with the component index to generate random samples, that is, elements
``(k, \\phi)`` are drawn from the augmented mixture model
```math
\\tilde{q}(\\phi, k | \\mu, \\Sigma) = K^{-1} q(\\phi | \\mu_k, \\Sigma_k),
```
where ``k`` is a component index, and ``K=`` `nruns`. These draws are then resampled with
replacement. Discarding ``k`` from the samples would reproduce draws from ``q``.

If `importance=true`, then Pareto smoothed importance resampling is used, so that the
resulting draws better approximate draws from the target distribution ``p`` instead of
``q``. This also prints a warning message if the importance weighted draws are unsuitable
for approximating expectations with respect to ``p``.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
    automatically differentiated using the backend specified in `ad_backend`.
- `fun::GalacticOptim.OptimizationFunction`: an optimization function that represents
    `-logp(x)` with its gradient. It must have the necessary features (e.g. a Hessian
    function) for the chosen optimization algorithm. For details, see
    [GalacticOptim.jl: OptimizationFunction](https://galacticoptim.sciml.ai/stable/API/optimization_function/).
- `ndraws::Int`: number of approximate draws to return

# Keywords
- `init`: iterator of length `nruns` of initial points of length `dim` from which each
    single-path Pathfinder run will begin. `length(init)` must be implemented. If `init` is
    not provided, `dim` and `nruns` must be.
- `nruns::Int`: number of runs of Pathfinder to perform. Ignored if `init` is provided.
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend used to
    differentiate `logp` if `∇logp` is not provided.
- `ndraws_per_run::Int`: The number of draws to take for each component before resampling.
    Defaults to a number such that `ndraws_per_run * nruns > ndraws`.
- `importance::Bool=true`: Perform Pareto smoothed importance resampling of draws.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Pseudorandom number generator. It is recommended to
    use a parallelization-friendly PRNG like the default PRNG on Julia 1.7 and up.
- `executor::Transducers.Executor`: Transducers.jl executor that determines if and how
    to run the single-path runs in parallel. If `rng` is known to be thread-safe, the
    default is `Transducers.PreferParallel(; basesize=1)` (parallel executation, defaulting
    to multi-threading). Otherwise, it is `Transducers.SequentialEx()` (no parallelization).
- `executor_per_run::Transducers.Executor=Transducers.SequentialEx()`: Transducers.jl
    executor used within each run to parallelize PRNG calls. Defaults to no parallelization.
    See [`pathfinder`](@ref) for a description.
- `kwargs...` : Remaining keywords are forwarded to [`pathfinder`](@ref).

# Returns
- `q::Distributions.MixtureModel`: Uniformly weighted mixture of ELBO-maximizing
    multivariate normal distributions
- `ϕ::AbstractMatrix{<:Real}`: approximate draws from target distribution with size
    `(dim, ndraws)`
- `component_inds::Vector{Int}`: Indices ``k`` of components in ``q`` from which each column
    in `ϕ` was drawn.
"""
function multipathfinder end

function multipathfinder(logp, ndraws::Int; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return multipathfinder(build_optim_function(logp; ad_backend), ndraws; kwargs...)
end
function multipathfinder(
    logp, ∇logp, ndraws::Int; ad_backend=AD.ForwardDiffBackend(), kwargs...
)
    return multipathfinder(build_optim_function(logp, ∇logp; ad_backend), ndraws; kwargs...)
end
function multipathfinder(
    optim_fun::GalacticOptim.OptimizationFunction,
    ndraws::Int;
    init=nothing,
    nruns::Int=init === nothing ? -1 : length(init),
    ndraws_elbo::Int=DEFAULT_NDRAWS_ELBO,
    ndraws_per_run::Int=max(ndraws_elbo, cld(ndraws, max(nruns, 1))),
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    executor::Transducers.Executor=_default_executor(rng; basesize=1),
    executor_per_run=Transducers.SequentialEx(),
    importance::Bool=true,
    kwargs...,
)
    if optim_fun.grad === nothing || optim_fun.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    if init === nothing
        nruns > 0 || throw(
            ArgumentError("A positive `nruns` must be set or `init` must be provided.")
        )
        _init = fill(init, nruns)
    else
        _init = init
    end
    if ndraws > ndraws_per_run * nruns
        @warn "More draws requested than total number of draws across replicas. Draws will not be unique."
    end
    logp(x) = -optim_fun.f(x, nothing)

    # run pathfinder independently from each starting point
    trans = Transducers.Map() do (init_i)
        return pathfinder(
            optim_fun;
            rng,
            ndraws=ndraws_per_run,
            init=init_i,
            executor=executor_per_run,
            ndraws_elbo,
            kwargs...,
        )
    end
    iter_sp = Transducers.withprogress(_init; interval=1e-3) |> trans
    pathfinder_results = Folds.collect(iter_sp, executor)
    fit_dist_opts = pathfinder_results |> Transducers.Map(x -> x.fit_dist_opt) |> collect
    draws_all = reduce(hcat, pathfinder_results |> Transducers.Map(x -> x.draws))

    # draw samples from augmented mixture model
    inds = axes(draws_all, 2)
    sample_inds, psis_result = if importance
        log_densities_fit =
            pathfinder_results |>
            Transducers.MapCat() do x
                return Distributions.logpdf(x.fit_dist_opt, x.draws)
            end |>
            collect
        iter_logp = eachcol(draws_all) |> Transducers.Map(logp)
        log_densities_target = Folds.collect(iter_logp, executor)
        log_densities_ratios = log_densities_target - log_densities_fit
        resample(rng, inds, log_densities_ratios, ndraws)
    else
        resample(rng, inds, ndraws), nothing
    end

    fit_dist_mix = Distributions.MixtureModel(fit_dist_opts)
    draws = draws_all[:, sample_inds]

    # get component ids (k) of draws in ϕ
    draw_component_ids = cld.(sample_inds, ndraws_per_run)

    return (; fit_dist_mix, draws, draw_component_ids, pathfinder_results, psis_result)
end
