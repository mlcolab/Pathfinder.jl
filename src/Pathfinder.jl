module Pathfinder

using Distributions: MixtureModel, MvNormal
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
    L = length(θs) - 1
    @assert length(logpθs) == length(∇logpθs) == L + 1

    # fit mv-normal distributions to trajectory
    dists = fit_mvnormal(θs, ∇logpθs; history_length=history_length)

    # find ELBO-maximizing distribution
    ϕ_logqϕ_λ = map(dists) do dist
        ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws_elbo)
        λ = elbo(logp.(ϕ), logqϕ)
        return ϕ, logqϕ, λ
    end
    ϕ, logqϕ, λ = ntuple(i -> getindex.(ϕ_logqϕ_λ, i), Val(3))
    lopt = argmax(λ[2:end]) + 1
    @info "Optimized for $L iterations. Maximum ELBO of $(round(λ[lopt]; digits=2)) reached at iteration $(lopt - 1)."

    # get parameters of ELBO-maximizing distribution
    distopt = dists[lopt]

    # reuse existing draws; draw additional ones if necessary
    ϕdraws = ϕ[lopt]
    logqϕdraws = logqϕ[lopt]
    if ndraws_elbo < ndraws
        append!.((ϕdraws, logqϕdraws), rand_and_logpdf(rng, distopt, ndraws - ndraws_elbo))
    else
        ϕdraws = ϕdraws[1:ndraws]
        logqϕdraws = logqϕdraws[1:ndraws]
    end

    return distopt, ϕdraws, logqϕdraws
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

    return MixtureModel(dists), ϕsample
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

elbo(logpϕ, logqϕ) = mean(logpϕ) - mean(logqϕ)

function psir(rng, ϕ, log_ratios, ndraws)
    log_weights, _ = PSIS.psis(log_ratios; normalize=true)
    weights = StatsBase.pweights(exp.(log_weights))
    return StatsBase.sample(rng, ϕ, weights, ndraws; replace=true)
end

# eq 4.9
# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function gilbert_initialization(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

nocedal_wright_scaling(α, s, y) = fill!(similar(α), dot(y, s) / sum(abs2, y))

"""
    fit_mvnormal(θs, ∇logpθs; cov_init=gilbert_initialization, history_length=5, ϵ=1e-12)

Fit a multivariate-normal distribution to each point on the trajectory `θs`.

Given `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian approximations with
the provided `history_length`. The inverse Hessians approximate a covariance. The
covariances and corresponding means that define multivariate normal approximations per
point are returned.
"""
function fit_mvnormal(
    θs, ∇logpθs; cov_init=gilbert_initialization, history_length=5, ϵ=1e-12
)
    L = length(θs) - 1
    θ = θs[1]
    ∇logpθ = ∇logpθs[1]

    # allocate caches/containers
    s = similar(θ) # BFGS update, i.e. sₗ = θₗ₊₁ - θₗ = -λ Hₗ ∇logpθₗ
    y = similar(∇logpθ) # cache for yₗ = ∇logpθₗ₊₁ - ∇logpθₗ = Hₗ₊₁ \ s₁ (secant equation)
    # (1)
    S = Vector{typeof(s)}(undef, 0)
    Y = Vector{typeof(y)}(undef, 0)
    α = fill!(similar(θ), true)
    Σ = lbfgs_inverse_hessian(α, S, Y) # Σ₀ = I
    μ = muladd(Σ, ∇logpθ, θ)
    dists = [MvNormal(μ, Σ)]

    # (2)
    for l in 1:L
        # (b)
        s .= θs[l + 1] .- θ
        y .= ∇logpθ .- ∇logpθs[l + 1]
        # (d)
        if dot(y, s) > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            # (i)
            push!(S, copy(s))
            push!(Y, copy(y))

            # (ii)
            # replace oldest stored s and y with new ones
            if length(S) > history_length
                s = popfirst!(S)
                y = popfirst!(Y)
            end

            # (iii-iv)
            # initial diagonal estimate of Σ
            α = cov_init(α, s, y)
        else
            @warn "Skipping inverse Hessian update to avoid negative curvature."
        end

        # (a)
        Σ = lbfgs_inverse_hessian(α, S, Y)
        θ = θs[l + 1]
        ∇logpθ = ∇logpθs[l + 1]
        μ = muladd(Σ, ∇logpθ, θ)
        push!(dists, MvNormal(μ, Σ))
    end
    return dists
end

# faster than computing `logpdf` and `rand` independently
function rand_and_logpdf(rng, dist::MvNormal{T,<:WoodburyPDMat{T}}, ndraws) where {T}
    μ = dist.μ
    Σ = dist.Σ
    N = length(μ)

    # draw points
    u = Random.randn!(rng, similar(μ, N, ndraws))
    unormsq = map(x -> sum(abs2, x), eachcol(u))
    x = PDMats.unwhiten!(u, Σ, u)
    x .+= μ

    # compute log density at each point
    logpx = ((logdet(Σ) + N * log2π) .+ unormsq) ./ -2

    return collect(eachcol(x)), logpx
end

end
