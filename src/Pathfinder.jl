module Pathfinder

using Distributions: Distributions
using LinearAlgebra
using Optim: Optim, LineSearches
using PDMats: PDMats
using PSIS
using Random
using Statistics: mean
using StatsBase: StatsBase
using StatsFuns: log2π

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_OPTIMIZER = Optim.LBFGS(; m=5, linesearch=LineSearches.MoreThuente())

include("woodbury.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")

"""
    pathfinder(logp, ∇logp, θ₀::AbstractVector{<:Real}, ndraws::Int; kwargs...)

Find the best multivariate normal approximation encountered while optimizing `logp`.

The multivariate normal approximation returned is the one that maximizes the evidence lower
bound (ELBO), or equivalently, minimizes the KL divergence between

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`.
- `θ₀`: initial point from which to begin optimization
- `ndraws`: number of approximate draws to return

# Keywords
- `rng::Random.AbstractRNG`: The random number generator to be used for drawing samples
- `optimizer::Optim.AbstractOptimizer`: Optimizer to be used for constructing trajectory.
    Defaults to `Optim.LBFGS(; m=5, linesearch=LineSearches.MoreThuente())`.
- `history_length::Int=5`: Size of the history used to approximate the inverse Hessian.
    This should only be set when `optimizer` is not an `Optim.LBFGS`.
- `ndraws_elbo::Int=5`: Number of draws used to estimate the ELBO
- `kwargs...` : Remaining keywords are forwarded to `Optim.Options`.

# Returns
- `dist::Distributions.MvNormal`: ELBO-maximizing multivariate normal distribution
- `ϕ::Vector{<:AbstractVector{<:Real}}`: `ndraws` draws from multivariate normal
- `logqϕ::Vector{<:Real}`: log-density of multivariate normal at `ϕ` values
"""
function pathfinder(
    logp,
    ∇logp,
    θ₀,
    ndraws;
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer::Optim.AbstractOptimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : 5,
    ndraws_elbo::Int=5,
    kwargs...,
)
    # compute trajectory
    θs, logpθs, ∇logpθs = optimize(logp, ∇logp, θ₀, optimizer; kwargs...)
    @assert length(logpθs) == length(∇logpθs)
    L = length(θs) - 1

    # fit mv-normal distributions to trajectory
    dists = fit_mvnormals(θs, ∇logpθs; history_length=history_length)

    # find ELBO-maximizing distribution
    lopt, ϕ, logqϕ, λ = maximize_elbo(rng, logp, dists[2:end], ndraws_elbo)
    @info "Optimized for $L iterations. Maximum ELBO of $(round(λ[lopt]; digits=2)) reached at iteration $lopt."

    # get parameters of ELBO-maximizing distribution
    distopt = dists[lopt + 1]

    # reuse existing draws; draw additional ones if necessary
    ϕopt = copy(ϕ[lopt])
    logqϕopt = copy(logqϕ[lopt])
    if ndraws_elbo < ndraws
        ϕnew, logqϕnew = rand_and_logpdf(rng, distopt, ndraws - ndraws_elbo)
        append!(ϕopt, ϕnew)
        append!(logqϕopt, logqϕnew)
    end

    return distopt, ϕopt, logqϕopt
end

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
Draws from the components are then resampled with replacement. If `filter=true`, then
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
- `dist::Distributions.MixtureModel`: Uniformly weighted mixture of ELBO-maximizing
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
    # TODO: allow to be parallelized
    res = map(θ₀s) do θ₀
        return pathfinder(logp, ∇logp, θ₀, ndraws_per_run; rng=rng, kwargs...)
    end
    dists, ϕs, logqϕs = ntuple(i -> getindex.(res, i), Val(3))

    # draw samples from mixture of multivariate normal distributions
    ϕsvec = reduce(vcat, ϕs)
    ϕsample = if importance
        # perform importance resampling
        log_ratios = logp.(ϕsvec) .- reduce(vcat, logqϕs)
        psir(rng, ϕsvec, log_ratios, ndraws)
    else
        StatsBase.sample(rng, ϕsvec, ndraws; replace=true)
    end

    return Distributions.MixtureModel(dists), ϕsample
end

function optimize(logp, ∇logp, θ₀, optimizer; kwargs...)
    f(x) = -logp(x)
    g!(y, x) = (y .= .-∇logp(x))

    options = Optim.Options(; store_trace=true, extended_trace=true, kwargs...)
    res = Optim.optimize(f, g!, θ₀, optimizer, options)

    θ = Optim.minimizer(res)
    θs = Optim.x_trace(res)::Vector{typeof(θ)}
    logpθs = -Optim.f_trace(res)
    ∇logpθs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))::typeof(θs)

    return θs, logpθs, ∇logpθs
end

function psir(rng, ϕ, log_ratios, ndraws)
    log_weights, _ = PSIS.psis(log_ratios; normalize=true)
    weights = StatsBase.pweights(exp.(log_weights))
    return StatsBase.sample(rng, ϕ, weights, ndraws; replace=true)
end

end
