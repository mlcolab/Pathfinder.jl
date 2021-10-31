"""
    multipathfinder(
        logp,
        ∇logp,
        θ₀s::AbstractVector{AbstractVector{<:Real}},
        ndraws::Int;
        kwargs...
    )

Filter samples from a mixture of multivariate normal distributions fit using `pathfinder`.

For `n=length(θ₀s)`, `n` parallel runs of pathfinder produce `n` multivariate normal
approximations of the posterior. These are combined to a mixture model with uniform weights.

Draws from the components are then resampled with replacement. If `importance=true`, then
Pareto smoothed importance resampling is used, so that the resulting draws better
approximate draws from the target distribution.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`.
- `θ₀s`: vector of initial points from which each optimization will begin
- `ndraws`: number of approximate draws to return

# Keywords
- `ndraws_per_run::Int=5`: The number of draws to take for each component before resampling.
- `importance::Bool=true`: Perform Pareto smoothed importance resampling of draws.

# Returns
- `q::Distributions.MixtureModel`: Uniformly weighted mixture of ELBO-maximizing
    multivariate normal distributions
- `ϕ::Vector{<:AbstractVector{<:Real}}`: `ndraws` approxiate draws from target distribution
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
    # execute remaining runs in parallel
    Threads.@threads for i in 2:nruns
        θ₀ = θ₀s[i]
        qs[i], ϕs[i], logqϕs[i] = pathfinder(
            logp, ∇logp, θ₀, ndraws_per_run; rng=rng, kwargs...
        )
    end

    # draw samples from mixture of multivariate normal distributions
    ϕsvec = reduce(vcat, ϕs)
    ϕsample = if importance
        log_ratios = logp.(ϕsvec) .- reduce(vcat, logqϕs)
        resample(rng, ϕsvec, log_ratios, ndraws)
    else
        resample(rng, ϕsvec, ndraws)
    end

    return Distributions.MixtureModel(qs), ϕsample
end
