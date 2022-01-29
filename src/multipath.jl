"""
    multipathfinder(
        logp,
        [∇logp,]
        θ₀s::AbstractVector{AbstractVector{<:Real}},
        ndraws::Int;
        kwargs...
    )

Filter samples from a mixture of multivariate normal distributions fit using `pathfinder`.

For `nruns=length(θ₀s)`, `nruns` parallel runs of pathfinder produce `nruns` multivariate
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
``q``.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
    automatically differentiated using the backend specified in `ad_backend`.
- `θ₀s::AbstractVector{AbstractVector{<:Real}}`: vector of length `nruns` of initial points
    of length `dim` from which each single-path Pathfinder run will begin
- `ndraws::Int`: number of approximate draws to return

# Keywords
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend.
- `ndraws_per_run::Int=5`: The number of draws to take for each component before resampling.
- `importance::Bool=true`: Perform Pareto smoothed importance resampling of draws.
- `kwargs...` : Remaining keywords are forwarded to [`pathfinder`](@ref).

# Returns
- `q::Distributions.MixtureModel`: Uniformly weighted mixture of ELBO-maximizing
    multivariate normal distributions
- `ϕ::AbstractMatrix{<:Real}`: approximate draws from target distribution with size
    `(dim, ndraws)`
- `component_inds::Vector{Int}`: Indices ``k`` of components in ``q`` from which each column
    in `ϕ` was drawn.
"""
function multipathfinder(logp, θ₀s, ndraws; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    optim_fun = build_optim_function(logp; ad_backend)
    return multipathfinder(optim_fun, θ₀s, ndraws; kwargs...)
end
function multipathfinder(
    logp, ∇logp, θ₀s, ndraws; ad_backend=AD.ForwardDiffBackend(), kwargs...
)
    optim_fun = build_optim_function(logp, ∇logp; ad_backend)
    return multipathfinder(optim_fun, θ₀s, ndraws; kwargs...)
end

"""
    multipathfinder(
        f::GalacticOptim.OptimizationFunction,
        θ₀s::AbstractVector{<:Real},
        ndraws::Int;
        kwargs...,
    )

Filter samples from a mixture of multivariate normal distributions fit using `pathfinder`.

`f` is a user-created optimization function that represents the negative log density with
its gradient and must have the necessary features (e.g. a Hessian function or specified
automatic differentiation type) for the chosen optimization algorithm. For details, see
[GalacticOptim.jl: OptimizationFunction](https://galacticoptim.sciml.ai/stable/API/optimization_function/).

See [`multipathfinder`](@ref) for a description of remaining arguments.
"""
function multipathfinder(
    optim_fun::GalacticOptim.OptimizationFunction,
    θ₀s,
    ndraws;
    ndraws_per_run::Int=5,
    rng::Random.AbstractRNG=Random.default_rng(),
    importance::Bool=true,
    kwargs...,
)
    if optim_fun.grad === nothing || optim_fun.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    if ndraws > ndraws_per_run * length(θ₀s)
        @warn "More draws requested than total number of draws across replicas. Draws will not be unique."
    end
    logp(x) = -optim_fun.f(x, nothing)

    # run pathfinder independently from each starting point
    # TODO: allow to be parallelized
    res = map(θ₀s) do θ₀
        return pathfinder(optim_fun, θ₀, ndraws_per_run; rng, kwargs...)
    end
    qs = reduce(vcat, first.(res))
    ϕs = reduce(hcat, getindex.(res, 2))

    # draw samples from augmented mixture model
    inds = axes(ϕs, 2)
    sample_inds = if importance
        logqϕs = reduce(vcat, last.(res))
        log_ratios = map(((ϕ, logqϕ),) -> logp(ϕ) - logqϕ, zip(eachcol(ϕs), logqϕs))
        resample(rng, inds, log_ratios, ndraws)
    else
        resample(rng, inds, ndraws)
    end

    q = Distributions.MixtureModel(qs)
    ϕ = ϕs[:, sample_inds]

    # get component ids (k) of draws in ϕ
    component_ids = cld.(sample_inds, ndraws_per_run)

    return q, ϕ, component_ids
end
