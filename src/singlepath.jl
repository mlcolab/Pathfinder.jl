
"""
    PathfinderResult

Container for results of single-path Pathfinder.

# Fields
- `input`: User-provided input object, e.g. either `logp`, `(logp, ∇logp)`, `optim_fun`,
    `optim_prob`, or another object.
- `optimizer`: Optimizer used for maximizing the log-density
- `rng`: Pseudorandom number generator that was used for sampling
- `optim_prob::GalacticOptim.OptimizationProblem`: Otimization problem used for
    optimization
- `logp`: Log-density function
- `fit_distribution::Distributions.MvNormal`: ELBO-maximizing multivariate normal distribution
- `draws::AbstractMatrix{<:Real}`: draws from multivariate normal with size `(dim, ndraws)`
- `fit_distribution_transformed`: `fit_distribution` transformed to the same space as the user-
    supplied target distribution. This is only different from `fit_distribution` when
    integrating with other packages, and its type depends on the type of `input`.
- `draws_transformed`: `draws` transformed to be draws from `fit_distribution_transformed`.
- `iteration_opt::Int`: Iteration at which ELBO estimate was maximized
- `num_tries::Int`: Number of tries until Pathfinder succeeded
- `optim_solution::GalacticOptim.OptimizationSolution`: Solution object of optimization.
- `optim_trace::Pathfinder.OptimizationTrace`: container for optimization trace of points,
    log-density, and gradient. The first point is the initial point.
- `fit_distributions::AbstractVector{Distributions.MvNormal}`: Multivariate normal distributions
    for each point in `optim_trace`, where `fit_distributions[iteration_opt + 1] == fit_distribution`
- `elbo_estimates::AbstractVector{<:Pathfinder.ELBOEstimate}`: ELBO estimates for all but
    the first distribution in `fit_distributions`.

# Returns
- [`PathfinderResult`](@ref)
"""
struct PathfinderResult{I,O,R,OP,LP,FD,D,FDT,DT,OS,OT,EE}
    input::I
    optimizer::O
    rng::R
    optim_prob::OP
    logp::LP
    fit_distribution::FD
    draws::D
    fit_distribution_transformed::FDT
    draws_transformed::DT
    iteration_opt::Int
    num_tries::Int
    optim_solution::OS
    optim_trace::OT
    fit_distributions::Vector{FD}
    elbo_estimates::EE
end

function Base.show(io::IO, ::MIME"text/plain", result::PathfinderResult)
    println(io, "Single-path Pathfinder result")
    println(io, "  tries: $(result.num_tries)")
    println(io, "  draws: $(size(result.draws, 1))")
    println(
        io, "  fit iteration: $(result.iteration_opt) / $(length(result.optim_trace) - 1)"
    )
    println(io, "  fit ELBO: $(_to_string(result.elbo_estimates[result.iteration_opt]))")
    print(io, "  fit distribution: ", result.fit_distribution)
    return nothing
end

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
- [`PathfinderResult`](@ref)
"""
function pathfinder end

function pathfinder(logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(build_optim_function(logp; ad_backend); input=logp, kwargs...)
end
function pathfinder(logp, ∇logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(
        build_optim_function(logp, ∇logp; ad_backend); input=(logp, ∇logp), kwargs...
    )
end
function pathfinder(
    optim_fun::GalacticOptim.OptimizationFunction;
    rng=Random.GLOBAL_RNG,
    init=nothing,
    dim::Int=-1,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    input=optim_fun,
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
    return pathfinder(prob; rng, input, init_sampler, allow_mutating_init, kwargs...)
end
function pathfinder(
    prob::GalacticOptim.OptimizationProblem;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    optimizer=DEFAULT_OPTIMIZER,
    ndraws_elbo::Int=DEFAULT_NDRAWS_ELBO,
    ndraws::Int=ndraws_elbo,
    input=prob,
    kwargs...,
)
    if prob.f.grad === nothing || prob.f.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    logp(x) = -prob.f.f(x, nothing)
    path_result = ProgressLogging.progress(; name="Optimizing") do progress_id
        return _pathfinder_try_until_succeed(
            rng, prob, logp; optimizer, progress_id, ndraws_elbo, kwargs...
        )
    end
    @unpack (
        itry,
        success,
        optim_solution,
        optim_trace,
        fit_distributions,
        iteration_opt,
        elbo_estimates,
    ) = path_result

    if !success
        ndraws_elbo_actual = 0
        @warn "Pathfinder failed after $itry tries. Increase `ntries`, inspect the model for numerical instability, or provide a more suitable `init_sampler`."
    else
        elbo_estimate_opt = elbo_estimates[iteration_opt]
        ndraws_elbo_actual = ndraws_elbo
    end

    fit_distribution = fit_distributions[iteration_opt + 1]

    # reuse existing draws; draw additional ones if necessary
    draws = if ndraws_elbo_actual == 0
        rand(rng, fit_distribution, ndraws)
    elseif ndraws_elbo_actual < ndraws
        hcat(elbo_estimate_opt.draws, rand(rng, fit_distribution, ndraws - ndraws_elbo_actual))
    else
        elbo_estimate_opt.draws[:, 1:ndraws]
    end

    # placeholders
    fit_distribution_transformed = fit_distribution
    draws_transformed = draws

    return PathfinderResult(
        input,
        optimizer,
        rng,
        optim_solution.prob,
        logp,
        fit_distribution,
        draws,
        fit_distribution_transformed,
        draws_transformed,
        iteration_opt,
        itry,
        optim_solution,
        optim_trace,
        fit_distributions,
        elbo_estimates,
    )
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
    fit_distributions = fit_mvnormals(
        optim_trace.points, optim_trace.gradients; history_length
    )

    # find ELBO-maximizing distribution
    iteration_opt, elbo_estimates = @views maximize_elbo(
        rng, logp, fit_distributions[(begin + 1):end], ndraws_elbo, executor
    )
    if isempty(elbo_estimates)
        success = false
    else
        elbo = elbo_estimates[iteration_opt].value
        success &= !isnan(elbo) & (elbo != -Inf)
    end

    return (;
        success,
        optim_solution,
        optim_trace,
        fit_distributions,
        iteration_opt,
        elbo_estimates,
    )
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
