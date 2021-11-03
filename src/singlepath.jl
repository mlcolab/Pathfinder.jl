"""
    pathfinder(logp, ∇logp, θ₀::AbstractVector{<:Real}, ndraws::Int; kwargs...)

Find the best multivariate normal approximation encountered while optimizing `logp`.

The multivariate normal approximation returned is the one that maximizes the evidence lower
bound (ELBO), or equivalently, minimizes the KL divergence between

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`.
- `θ₀`: initial point of length `dim` from which to begin optimization
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
- `q::Distributions.MvNormal`: ELBO-maximizing multivariate normal distribution
- `ϕ::AbstractMatrix{<:Real}`: draws from multivariate normal with size `(dim, ndraws)`
- `logqϕ::Vector{<:Real}`: log-density of multivariate normal at columns of `ϕ`
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
    θs, logpθs, ∇logpθs = maximize_with_trace(logp, ∇logp, θ₀, optimizer; kwargs...)
    L = length(θs) - 1
    @assert L + 1 == length(logpθs) == length(∇logpθs)

    # fit mv-normal distributions to trajectory
    qs = fit_mvnormals(θs, ∇logpθs; history_length=history_length)

    # find ELBO-maximizing distribution
    lopt, elbo, ϕ, logqϕ = maximize_elbo(rng, logp, qs[2:end], ndraws_elbo)
    @info "Optimized for $L iterations. Maximum ELBO of $(round(elbo; digits=2)) reached at iteration $lopt."

    # get parameters of ELBO-maximizing distribution
    q = qs[lopt + 1]

    # reuse existing draws; draw additional ones if necessary
    if ndraws_elbo < ndraws
        ϕnew, logqϕnew = rand_and_logpdf(rng, q, ndraws - ndraws_elbo)
        ϕ = hcat(ϕ, ϕnew)
        append!(logqϕ, logqϕnew)
    elseif ndraws < ndraws_elbo
        ϕ = ϕ[:, 1:ndraws]
        logqϕ = logqϕ[1:ndraws]
    end

    return q, ϕ, logqϕ
end
