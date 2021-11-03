"""
    multipathfinder(
        logp,
        ∇logp,
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
- `∇logp`: a callable that computes the gradient of `logp`.
- `θ₀s`: vector of length `nruns` of initial points of length `dim` from which each
    single-path Pathfinder run will begin
- `ndraws`: number of approximate draws to return

# Keywords
- `ndraws_per_run::Int=5`: The number of draws to take for each component before resampling.
- `importance::Bool=true`: Perform Pareto smoothed importance resampling of draws.

# Returns
- `q::Distributions.MixtureModel`: Uniformly weighted mixture of ELBO-maximizing
    multivariate normal distributions
- `ϕ::AbstractMatrix{<:Real}`: approximate draws from target distribution with size
    `(dim, ndraws)`
- `component_inds::Vector{Int}`: Indices ``k`` of components in ``q`` from which each column
in `ϕ` was drawn.
"""
function multipathfinder(
    logp,
    ∇logp,
    θ₀s,
    ndraws;
    ndraws_per_run::Int=5,
    rng::Random.AbstractRNG=Random.default_rng(),
    importance::Bool=true,
    kwargs...,
)
    if ndraws > ndraws_per_run * length(θ₀s)
        @warn "More draws requested than total number of draws across replicas. Draws will not be unique."
    end

    # run pathfinder independently from each starting point
    nruns = length(θ₀s)
    # run pathfinder once to get output types
    q₁, ϕ₁, logqϕ₁ = pathfinder(logp, ∇logp, first(θ₀s), ndraws_per_run; rng=rng, kwargs...)
    qs = Vector{typeof(q₁)}(undef, nruns)
    ϕs = Vector{typeof(ϕ₁)}(undef, nruns)
    logqϕs = Vector{typeof(logqϕ₁)}(undef, nruns)
    qs[1], ϕs[1], logqϕs[1] = q₁, ϕ₁, logqϕ₁
    if nruns > 1
        interval = 1:min(nruns - 1, Threads.nthreads())
        # deepcopy logp and ∇logp in case it's a callable that mutates inner state
        logps = [deepcopy(logp) for _ in interval]
        ∇logps = [deepcopy(∇logp) for _ in interval]
        # copy of RNG for each thread
        rngs = [deepcopy(rng) for _ in interval]
        # individual seeds for each run
        seeds = rand(rng, UInt, nruns - 1)

        Threads.@threads for i in 2:nruns
            id = Threads.threadid()
            rngᵢ = rngs[id]
            Random.seed!(rngᵢ, seeds[i - 1])
            qs[i], ϕs[i], logqϕs[i] = pathfinder(
                logps[id], ∇logps[id], θ₀s[i], ndraws_per_run; rng=rngᵢ, kwargs...
            )
        end
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
