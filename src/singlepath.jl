"""
    pathfinder(logp[, ∇logp], θ₀::AbstractVector{<:Real}, ndraws::Int; kwargs...)

Find the best multivariate normal approximation encountered while minimizing `logp`.

The multivariate normal approximation returned is the one that maximizes the evidence lower
bound (ELBO), or equivalently, minimizes the KL divergence between

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
automatically differentiated using the backend specified in `ad_backend`.
- `θ₀`: initial point of length `dim` from which to begin optimization
- `ndraws`: number of approximate draws to return

# Keywords
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend.
- `rng::Random.AbstractRNG`: The random number generator to be used for drawing samples
- `optimizer`: Optimizer to be used for constructing trajectory. Can be any optimizer
compatible with GalacticOptim, so long as it supports callbacks. Defaults to
`Optim.LBFGS(; m=$DEFAULT_HISTORY_LENGTH, linesearch=LineSearches.MoreThuente())`. See the
[GalacticOptim.jl documentation](https://galacticoptim.sciml.ai/stable) for details.
- `history_length::Int=$DEFAULT_HISTORY_LENGTH`: Size of the history used to approximate the
inverse Hessian. This should only be set when `optimizer` is not an `Optim.LBFGS`.
- `ndraws_elbo::Int=5`: Number of draws used to estimate the ELBO
- `kwargs...` : Remaining keywords are forwarded to `GalacticOptim.OptimizationProblem`.

# Returns
- `q::Distributions.MvNormal`: ELBO-maximizing multivariate normal distribution
- `ϕ::AbstractMatrix{<:Real}`: draws from multivariate normal with size `(dim, ndraws)`
- `logqϕ::Vector{<:Real}`: log-density of multivariate normal at columns of `ϕ`
"""
function pathfinder(logp, ∇logp, θ₀, ndraws; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    optim_fun = build_optim_function(logp, ∇logp; ad_backend)
    return pathfinder(optim_fun, θ₀, ndraws; kwargs...)
end
function pathfinder(logp, θ₀, ndraws; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    optim_fun = build_optim_function(logp; ad_backend)
    return pathfinder(optim_fun, θ₀, ndraws; kwargs...)
end

"""
    pathfinder(
        f::GalacticOptim.OptimizationFunction,
        θ₀::AbstractVector{<:Real},
        ndraws::Int;
        kwargs...,
    )

Find the best multivariate normal approximation encountered while maximizing `f`.

`f` is a user-created optimization function that represents the negative log density with
its gradient and must have the necessary features (e.g. a Hessian function or specified
automatic differentiation type) for the chosen optimization algorithm. For details, see
[GalacticOptim.jl: OptimizationFunction](https://galacticoptim.sciml.ai/stable/API/optimization_function/).

See [`pathfinder`](@ref) for a description of remaining arguments.
"""
function pathfinder(
    optim_fun::GalacticOptim.OptimizationFunction,
    θ₀,
    ndraws;
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : DEFAULT_HISTORY_LENGTH,
    ndraws_elbo::Int=5,
    kwargs...,
)
    optim_prob = build_optim_problem(optim_fun, θ₀; kwargs...)
    return pathfinder(
        optim_prob,
        ndraws;
        rng=rng,
        optimizer=optimizer,
        history_length=history_length,
        ndraws_elbo=ndraws_elbo,
    )
end

"""
    pathfinder(prob::GalacticOptim.OptimizationProblem, ndraws::Int; kwargs...)

Find the best multivariate normal approximation encountered while solving `prob`.

`prob` is a user-created optimization problem that represents the negative log density with
its gradient, an initial position and must have the necessary features (e.g. a Hessian
function or specified automatic differentiation type) for the chosen optimization algorithm.
For details, see
[GalacticOptim.jl: Defining OptimizationProblems](https://galacticoptim.sciml.ai/stable/API/optimization_problem/).

See [`pathfinder`](@ref) for a description of remaining arguments.
"""
function pathfinder(
    optim_prob::GalacticOptim.OptimizationProblem,
    ndraws;
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : DEFAULT_HISTORY_LENGTH,
    ndraws_elbo::Int=5,
)
    if optim_prob.f.grad === nothing || optim_prob.f.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    logp(x) = -optim_prob.f.f(x, Any)
    # compute trajectory
    θs, logpθs, ∇logpθs = optimize_with_trace(optim_prob, optimizer)
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
