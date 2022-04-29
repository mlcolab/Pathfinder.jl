"""
    pathfinder(logp; kwargs...)
    pathfinder(logp, ∇logp; kwargs...)
    pathfinder(fun::GalacticOptim::OptimizationFunction; kwargs...)

Find the best multivariate normal approximation encountered while maximizing `logp`.

From an optimization trajectory, Pathfinder constructs a sequence of (multivariate normal)
approximations to the distribution specified by `logp`. The approximation that maximizes the
evidence lower bound (ELBO), or equivalently, minimizes the KL divergence between the
approximation and the true distribution, is returned.

The covariance of the multivariate normal distribution is an inverse Hessian approximation
constructed using at most the previous `history_length` steps.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
    automatically differentiated using the backend specified in `ad_backend`.
- `fun::GalacticOptim.OptimizationFunction`: an optimization function that represents
    `-logp(x)` with its gradient. It must have the necessary features (e.g. a Hessian
    function) for the chosen optimization algorithm. For details, see
    [GalacticOptim.jl: OptimizationFunction](https://galacticoptim.sciml.ai/stable/API/optimization_function/).
- `prob::GalacticOptim.OptimizationProblem`: an optimization problem containing a function with
    the same properties as `fun`, as well as an initial point, in which case `init` and
    `dim` are ignored.

# Keywords
- `dim::Int`: dimension of the target distribution. If not provided, `init` or must be.
    Ignored if `init` is provided.
- `init::AbstractVector{<:Real}`: initial point of length `dim` from which to begin
    optimization. If not provided, an initial point of type `Vector{Float64}` and length
    `dim` is created and filled using `init_sampler`.
- `init_scale::Real`: scale factor ``s`` such that the default `init_sampler` samples
    entries uniformly in the range ``[-s, s]``
- `init_sampler`: function with the signature `(rng, x) -> x` that modifies a vector of
    length `dims` in-place to generate an initial point
- `ndraws_elbo::Int=$DEFAULT_NDRAWS_ELBO`: Number of draws used to estimate the ELBO
- `ndraws::Int=ndraws_elbo`: number of approximate draws to return
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend.
- `rng::Random.AbstractRNG`: The random number generator to be used for drawing samples
- `executor::Transducers.Executor=Transducers.SequentialEx()`: Transducers.jl executor that
    determines if and how to perform ELBO computation in parallel. The default
    (`SequentialEx()`) performs no parallelization. If `rng` is known to be thread-safe, and
    the log-density function is known to have no internal state, then
    `Transducers.PreferParallel()` may be used to parallelize log-density evaluation.
    This is generally only faster for expensive log density functions.
- `optimizer`: Optimizer to be used for constructing trajectory. Can be any optimizer
    compatible with GalacticOptim, so long as it supports callbacks. Defaults to
    `Optim.LBFGS(; m=$DEFAULT_HISTORY_LENGTH, linesearch=LineSearches.MoreThuente())`. See
    the [GalacticOptim.jl documentation](https://galacticoptim.sciml.ai/stable) for details.
- `history_length::Int=$DEFAULT_HISTORY_LENGTH`: Size of the history used to approximate the
    inverse Hessian. This should only be set when `optimizer` is not an `Optim.LBFGS`.
- `kwargs...` : Remaining keywords are forwarded to
    [`GalacticOptim.solve`](https://galacticoptim.sciml.ai/stable/API/solve).

# Returns
- `q::Distributions.MvNormal`: ELBO-maximizing multivariate normal distribution
- `ϕ::AbstractMatrix{<:Real}`: draws from multivariate normal with size `(dim, ndraws)`
- `logqϕ::Vector{<:Real}`: log-density of multivariate normal at columns of `ϕ`
"""
function pathfinder end

function pathfinder(logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(build_optim_function(logp; ad_backend); kwargs...)
end
function pathfinder(logp, ∇logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(build_optim_function(logp, ∇logp; ad_backend); kwargs...)
end
function pathfinder(
    optim_fun::GalacticOptim.OptimizationFunction;
    rng=Random.GLOBAL_RNG,
    init=nothing,
    dim::Int=-1,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    kwargs...,
)
    if init !== nothing
        _init = init
    elseif init === nothing && dim > 0
        _init = Vector{Float64}(undef, dim)
        init_sampler(rng, _init)
    else
        throw(ArgumentError("An initial point `init` or dimension `dim` must be provided."))
    end
    optim_prob = build_optim_problem(optim_fun, _init)
    return pathfinder(optim_prob; rng, kwargs...)
end
function pathfinder(
    optim_prob::GalacticOptim.OptimizationProblem;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    executor::Transducers.Executor=Transducers.SequentialEx(),
    optimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : DEFAULT_HISTORY_LENGTH,
    ndraws_elbo::Int=DEFAULT_NDRAWS_ELBO,
    ndraws::Int=ndraws_elbo,
    kwargs...,
)
    if optim_prob.f.grad === nothing || optim_prob.f.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    logp(x) = -optim_prob.f.f(x, nothing)
    # compute trajectory
    θs, logpθs, ∇logpθs = optimize_with_trace(optim_prob, optimizer; kwargs...)
    L = length(θs) - 1
    @assert L + 1 == length(logpθs) == length(∇logpθs)

    # fit mv-normal distributions to trajectory
    qs = fit_mvnormals(θs, ∇logpθs; history_length)

    # find ELBO-maximizing distribution
    lopt, elbo, ϕ, logqϕ = maximize_elbo(rng, logp, qs[2:end], ndraws_elbo, executor)
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

"""
    UniformSampler(scale::Real)

Sampler that in-place modifies an array to be IID uniformly distributed on `[-scale, scale]`
"""
struct UniformSampler{T<:Real}
    scale::T
    function UniformSampler(scale::T) where {T<:Real}
        scale > 0 || throw(DomainError(scale, "scale of uniform sampler must be positive."))
        return new{T}(scale)
    end
end

function (s::UniformSampler)(rng::Random.AbstractRNG, point)
    scale = s.scale
    @. point = rand(rng) * 2scale - scale
    return point
end
