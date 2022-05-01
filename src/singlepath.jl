"""
    pathfinder(logp; kwargs...)
    pathfinder(logp, ∇logp; kwargs...)
    pathfinder(fun::GalacticOptim::OptimizationFunction; kwargs...)
    pathfinder(prob::GalacticOptim::OptimizationProblem; kwargs...)

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
- `ntries::Int=1_000`: Number of times to try the optimization, restarting if it fails. Before
    every restart, a new initial point is drawn using `init_sampler`.
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
        allow_mutating_init = false
    elseif init === nothing && dim > 0
        _init = Vector{Float64}(undef, dim)
        init_sampler(rng, _init)
        allow_mutating_init = true
    else
        throw(ArgumentError("An initial point `init` or dimension `dim` must be provided."))
    end
    prob = build_optim_problem(optim_fun, _init)
    return pathfinder(prob; rng, init_sampler, allow_mutating_init, kwargs...)
end
function pathfinder(
    prob::GalacticOptim.OptimizationProblem;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    ndraws_elbo::Int=DEFAULT_NDRAWS_ELBO,
    ndraws::Int=ndraws_elbo,
    kwargs...,
)
    if prob.f.grad === nothing || prob.f.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    logp(x) = -prob.f.f(x, nothing)
    path_result = ProgressLogging.progress(; name="Optimizing") do progress_id
        return _pathfinder_try_until_succeed(
            rng, prob, logp; progress_id, ndraws_elbo, kwargs...
        )
    end
    @unpack itry, success, optim_solution, optim_trace, fit_dists, iteration_opt, elbo_estimates =
        path_result
    success ||
        @warn "Pathfinder failed after $itry tries. Increase `ntries`, inspect the model for numerical instability, or provide a more suitable `init_sampler`."

    # get parameters of ELBO-maximizing distribution
    elbo_estimate_opt = elbo_estimates[iteration_opt]
    fit_dist_opt = fit_dists[iteration_opt + 1]

    iterations = length(optim_trace) - 1
    @info "Optimized for $iterations iterations (tries: $itry). Maximum ELBO of $(_to_string(elbo_estimate_opt)) reached at iteration $iteration_opt."

    # reuse existing draws; draw additional ones if necessary
    draws = if ndraws_elbo < ndraws
        hcat(elbo_estimate_opt.draws, rand(rng, fit_dist_opt, ndraws - ndraws_elbo))
    else
        elbo_estimate_opt.draws[:, 1:ndraws]
    end

    return fit_dist_opt, draws
end

function _pathfinder_try_until_succeed(
    rng,
    prob,
    logp;
    ntries::Int=1_000,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    allow_mutating_init::Bool=false,
    kwargs...,
)
    itry = 1
    progress_name = "Optimizing (try 1)"
    result = _pathfinder(rng, prob, logp; progress_name, kwargs...)
    _prob = prob
    while !result.success && itry < ntries
        if itry == 1 && !allow_mutating_init
            _prob = deepcopy(prob)
        end
        itry += 1
        init_sampler(rng, _prob.u0)
        progress_name = "Optimizing (try $itry)"
        result = _pathfinder(rng, _prob, logp; progress_name, kwargs...)
    end
    return (; itry, result...)
end

function _pathfinder(
    rng,
    prob,
    logp;
    optimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : DEFAULT_HISTORY_LENGTH,
    ndraws_elbo=DEFAULT_NDRAWS_ELBO,
    executor::Transducers.Executor=Transducers.SequentialEx(),
    kwargs...,
)
    # compute trajectory
    optim_solution, optim_trace = optimize_with_trace(prob, optimizer; kwargs...)
    L = length(optim_trace) - 1
    success = L > 0

    # fit mv-normal distributions to trajectory
    fit_dists = fit_mvnormals(optim_trace.points, optim_trace.gradients; history_length)

    # find ELBO-maximizing distribution
    iteration_opt, elbo_estimates = @views maximize_elbo(rng, logp, fit_dists[begin+1:end], ndraws_elbo, executor)
    elbo = elbo_estimates[iteration_opt].value
    success &= !isnan(elbo) & (elbo != -Inf)

    return (; success, optim_solution, optim_trace, fit_dists, iteration_opt, elbo_estimates)
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
