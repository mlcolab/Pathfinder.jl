function build_optim_function(
    log_density_problem,
    adtype::ADTypes.AbstractADType,
    ::LogDensityProblems.LogDensityOrder{order};
) where {order}
    if order > 0
        function grad(res, x, _...)
            _, ∇fx = LogDensityProblems.logdensity_and_gradient(log_density_problem, x)
            @. res = -∇fx
            return res
        end
    else
        grad = nothing
    end
    if order > 1
        function hess(res, x, _...)
            _, _, H = LogDensityProblems.logdensity_gradient_and_hessian(
                log_density_problem, x
            )
            @. res = -H
            return res
        end
    else
        hess = nothing
    end
    return build_optim_function(
        Base.Fix1(LogDensityProblems.logdensity, log_density_problem), adtype; grad, hess
    )
end
function build_optim_function(log_density_fun, adtype::ADTypes.AbstractADType; kwargs...)
    f(x, _...) = -log_density_fun(x)
    return SciMLBase.OptimizationFunction(f, adtype; kwargs...)
end

function optimize_with_trace(
    prob,
    optimizer;
    progress_name="Optimizing",
    progress_id=nothing,
    maxiters=1_000,
    callback=nothing,
    fail_on_nonfinite=true,
    ndraws_elbo::Int=5,
    rng=Random.default_rng(),
    (invH_init!)=gilbert_invH_init!,
    save_trace::Bool=true,
    history_length::Int=DEFAULT_HISTORY_LENGTH,
    ntasks::Int=1,
    kwargs...,
)
    fun = _optim_function_with_gradient(prob, optimizer)

    logp(x) = -fun.f(x, nothing)
    function ∇logp(x)
        SciMLBase.isinplace(fun) || return -fun.grad(x)
        res = similar(x)
        fun.grad(res, x)
        rmul!(res, -1)
        return res
    end

    # caches for the trace of x and f(x)
    (; u0) = prob

    # TODO: keep deepcopy of ELBO-maximizing fit distribution so far, iteration where built,
    # and maximum ELBO value

    _callback = OptimizationCallback(
        logp,
        ∇logp,
        u0;
        ndraws_elbo,
        history_length,
        rng,
        save_trace,
        ntasks,
        maxiters,
        fail_on_nonfinite,
        callback,
        invH_init!,
        progress_name,
        progress_id,
    )
    sol = SciMLBase.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)

    (; optim_trace, fit_distributions, elbo_estimates, fit_distribution, fit_iteration) =
        _callback

    return (;
        optim_solution=sol,
        optim_trace,
        fit_distributions,
        elbo_estimates=elbo_estimates[(begin + 1):end],
        fit_distribution,
        fit_iteration,
    )
end

function _optim_function_with_gradient(prob, optimizer)
    prob.f.grad === nothing || return prob.f
    # Generate a cache to use OptimizationBase's native functionality for adding missing
    # gradient values
    fun = OptimizationBase.OptimizationCache(prob, optimizer).f
    if fun.grad === nothing
        throw(
            ArgumentError(
                "Gradient function is not available. Please provide an OptimizationProblem with an explicit gradient function.",
            ),
        )
    end
    return fun
end

function _fill_missing_gradient_values!(∇fxs, xs, optim_fun)
    function ∇f(x)
        SciMLBase.isinplace(optim_fun) || return optim_fun.grad(x)
        res = similar(x)
        optim_fun.grad(res, x)
        rmul!(res, -1)
        return res
    end
    map!(∇fxs, ∇fxs, xs) do ∇fx, x
        return ∇fx === nothing ? ∇f(x) : ∇fx
    end
    return convert(typeof(xs), ∇fxs)
end

mutable struct OptimizationCallback{
    F,
    DF,
    R<:Random.AbstractRNG,
    CB,
    L<:LBFGSState,
    DC<:AbstractMatrix,
    OT,
    FD<:Distributions.MvNormal,
    FDs<:Vector{<:FD},
    EE<:Vector,
    IH,
    ID,
}
    # Generated functions
    const logp::F
    const ∇logp::DF
    # User-provided options
    const rng::R
    const save_trace::Bool
    const ntasks::Int
    const maxiters::Int
    const fail_on_nonfinite::Bool
    const callback::CB
    # State/caches
    const lbfgs_state::L
    const draws_cache::DC
    const optim_trace::OT
    const fit_distributions::FDs
    const elbo_estimates::EE
    fit_distribution::FD
    fit_iteration::Int
    # Internally set options
    const invH_init!::IH
    const progress_name::String
    const progress_id::ID
end

# New constructor
function OptimizationCallback(
    logp,
    ∇logp,
    u0;
    rng::Random.AbstractRNG=Random.default_rng(),
    save_trace::Bool=true,
    ntasks::Int=1,
    maxiters::Int=1_000,
    fail_on_nonfinite::Bool=true,
    ndraws_elbo::Int=5,
    history_length::Int=DEFAULT_HISTORY_LENGTH,
    callback=nothing,
    (invH_init!)=gilbert_invH_init!,
    progress_name::String="Optimizing",
    progress_id=nothing,
)
    T = FT = eltype(u0)
    xs = typeof(u0)[]
    fxs = FT[]
    ∇fxs = typeof(u0)[]
    optim_trace = OptimizationTrace(xs, fxs, ∇fxs)
    lbfgs_state = LBFGSState(u0, zero(FT), zero(u0), history_length)
    draws_cache = similar(u0, size(u0, 1), ndraws_elbo)
    elbo_estimates = ELBOEstimate{T,typeof(draws_cache),Vector{T}}[]
    fit_iteration = -1
    fit_distribution = deepcopy(fit_mvnormal(lbfgs_state))
    fit_distributions = typeof(fit_distribution)[]

    return OptimizationCallback(
        logp,
        ∇logp,
        rng,
        save_trace,
        ntasks,
        maxiters,
        fail_on_nonfinite,
        callback,
        lbfgs_state,
        draws_cache,
        optim_trace,
        fit_distributions,
        elbo_estimates,
        fit_distribution,
        fit_iteration,
        invH_init!,
        progress_name,
        progress_id,
    )
end

function (cb::OptimizationCallback)(state::OptimizationState, args...)
    (;
        logp,
        ∇logp,
        rng,
        save_trace,
        ntasks,
        maxiters,
        fail_on_nonfinite,
        callback,
        lbfgs_state,
        optim_trace,
        fit_distributions,
        elbo_estimates,
        draws_cache,
        invH_init!,
        progress_name,
        progress_id,
    ) = cb
    ret = callback !== nothing && callback(state, args...)
    Base.@logmsg ProgressLogging.ProgressLevel progress_name progress = state.iter / maxiters _id =
        progress_id
    # optimizers differ in how they number iterations in `state.iter`, so we instead use the
    # index of the current point in the trace, where the initial point is iteration 0
    iteration = length(elbo_estimates)

    # some optimizers mutate x, so we must copy it
    x = copy(state.u)
    logp_x = -state.objective
    ∇logp_x = state.grad === nothing ? ∇logp(x) : -state.grad

    # Update L-BFGS state
    ϵ = sqrt(eps(eltype(x)))
    _update_state!(lbfgs_state, x, -logp_x, -∇logp_x, invH_init!, ϵ)

    # Fit distribution
    fit_distribution = fit_mvnormal(lbfgs_state)
    elbo_estimate = elbo_and_samples!(
        draws_cache, rng, logp, fit_distribution; save_samples=save_trace, ntasks
    )

    push!(optim_trace.log_densities, logp_x)
    push!(elbo_estimates, elbo_estimate)
    if save_trace
        push!(optim_trace.points, x)
        push!(optim_trace.gradients, ∇logp_x)
        push!(fit_distributions, deepcopy(fit_distribution))
    end

    # Keep track of the ELBO-maximizing iteration
    if cb.fit_iteration ≤ 0 || (
        isfinite(elbo_estimate.value) &&
        elbo_estimate.value > elbo_estimates[cb.fit_iteration + 1].value
    )
        cb.fit_iteration = iteration
        cb.fit_distribution = deepcopy(fit_distribution)
    end

    if fail_on_nonfinite && !ret
        ret = (
            isnan(logp_x) ||
            logp_x == Inf ||
            (∇logp_x !== nothing && any(!isfinite, ∇logp_x))
        )::Bool
    end

    return ret
end

struct OptimizationTrace{P,L}
    points::P
    log_densities::L
    gradients::P
end

Base.length(trace::OptimizationTrace) = length(trace.log_densities)

function Base.show(io::IO, trace::OptimizationTrace)
    print(io, "OptimizationTrace with $(length(trace) - 1) iterations")
    return nothing
end
