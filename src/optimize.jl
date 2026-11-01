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
    online_fit=nothing,
    kwargs...,
)
    u0 = prob.u0
    fun = prob.f
    # caches for the trace of x and f(x); with an online fit, the points and gradients are
    # consumed as they arrive and these two stay empty
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = Union{Nothing,typeof(u0)}[]
    _callback = OptimizationCallback(
        xs,
        fxs,
        ∇fxs,
        progress_name,
        progress_id,
        maxiters,
        callback,
        fail_on_nonfinite,
        online_fit,
    )
    sol = SciMLBase.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)

    _∇fxs = _fill_missing_gradient_values!(∇fxs, xs, sol.cache.f)

    return sol, OptimizationTrace(xs, fxs, _∇fxs)
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

struct OptimizationCallback{X,FX,∇FX,ID,CB,OF}
    xs::X
    fxs::FX
    ∇fxs::∇FX
    progress_name::String
    progress_id::ID
    maxiters::Int
    callback::CB
    fail_on_nonfinite::Bool
    online_fit::OF # `nothing`, or an `OnlineFit` that consumes the points and gradients
end

function (cb::OptimizationCallback)(state::OptimizationState, args...)
    (;
        xs,
        fxs,
        ∇fxs,
        progress_name,
        progress_id,
        maxiters,
        callback,
        fail_on_nonfinite,
        online_fit,
    ) = cb
    ret = callback !== nothing && callback(state, args...)
    iteration = state.iter
    Base.@logmsg ProgressLogging.ProgressLevel progress_name progress = iteration / maxiters _id =
        progress_id

    # some backends mutate x, so we must copy it
    x = copy(state.u)
    fx = -state.objective
    ∇fx = state.grad === nothing ? nothing : -state.grad

    push!(fxs, fx)
    if online_fit === nothing
        push!(xs, x)
        push!(∇fxs, ∇fx)
    else
        _fit!(online_fit, x, ∇fx)
    end

    if fail_on_nonfinite && !ret
        ret = (isnan(fx) || fx == Inf || (∇fx !== nothing && any(!isfinite, ∇fx)))::Bool
    end

    return ret
end

"""
    OnlineFit(prob::SciMLBase.OptimizationProblem, logp; rng, ndraws_elbo, history_length)

Fit the multivariate normal approximation at every point of the optimization trajectory as
the point arrives, estimate its ELBO, and keep only the ELBO-maximizing fit and its draws.

Passed as `online_fit` to [`optimize_with_trace`](@ref), it replaces storing the trajectory
and fitting after the optimization, so the memory it holds is independent of the number of
iterations: the L-BFGS state and two fits, `O(dim * history_length)`, plus the draws of the
selected fit, `O(dim * ndraws_elbo)`. Its `elbo_estimates` carry one entry per iteration
whose draws are kept only for `fit_iteration`, as `_without_draws` makes them.

The inverse Hessians are the ones [`lbfgs_inverse_hessians`](@ref) reconstructs from the
stored trajectory, built one point at a time through the same
[`LBFGSInverseHessianState`](@ref). Each ELBO is estimated right after its fit, from a
seed drawn from `rng` at that point, so it is reproducible for a given `rng` but uses a
different random stream than [`maximize_elbo`](@ref), which seeds all estimates after the
optimization. The gradient at a point comes from the optimizer; if it reports none, it is
computed from `prob.f`, which must then have a gradient function.
"""
mutable struct OnlineFit{
    T<:Real,
    L<:LBFGSInverseHessianState,
    LP,
    GF,
    R<:Random.AbstractRNG,
    RE<:Random.AbstractRNG,
    FD<:Distributions.MvNormal,
    EE<:ELBOEstimate,
}
    const lbfgs_state::L
    const logp::LP
    const ∇logp::GF # `nothing`, or `x -> ∇logp(x)`, for optimizers that report no gradient
    const rng::R # draws one seed per fitted point
    const elbo_rng::RE # re-seeded for every ELBO estimate
    const ndraws_elbo::Int
    const elbo_estimates::Vector{EE}
    fit_distribution::FD # the fit at `fit_iteration`
    fit_iteration::Int
    best::Union{Missing,Tuple{T,Int}} # `_findmax_skipnan` state over the ELBO values
    started::Bool # whether the first point has arrived
end

function OnlineFit(
    prob::SciMLBase.OptimizationProblem,
    logp;
    rng::Random.AbstractRNG,
    ndraws_elbo::Int,
    history_length::Int,
    Hinit=gilbert_init,
    ϵ=1e-12,
)
    (; u0) = prob
    ∇logp = _gradient_function(prob.f, prob.p)
    # placeholders until the first point arrives: they fix the types of the fits
    lbfgs_state = LBFGSInverseHessianState(u0, zero(u0); Hinit, history_length, ϵ)
    fit_distribution = fit_mvnormal(lbfgs_state)
    elbo_rng = copy(rng)
    EE = Base.promote_op(
        elbo_and_samples, typeof(elbo_rng), typeof(logp), typeof(fit_distribution), Int
    )
    T = float(eltype(u0))
    return OnlineFit{
        T,
        typeof(lbfgs_state),
        typeof(logp),
        typeof(∇logp),
        typeof(rng),
        typeof(elbo_rng),
        typeof(fit_distribution),
        EE,
    }(
        lbfgs_state,
        logp,
        ∇logp,
        rng,
        elbo_rng,
        ndraws_elbo,
        EE[],
        fit_distribution,
        0,
        missing,
        false,
    )
end

# `x -> ∇logp(x)` from the gradient function of the optimization function `optim_fun`, which
# computes the gradient of the NEGATIVE log density, or `nothing` if it has none.
function _gradient_function(optim_fun, p)
    optim_fun.grad === nothing && return nothing
    function ∇logp(x)
        SciMLBase.isinplace(optim_fun) || return -optim_fun.grad(x, p)
        res = similar(x)
        optim_fun.grad(res, x, p)
        rmul!(res, -1)
        return res
    end
    return ∇logp
end

# Advance the fit by the point `x` with gradient `∇logpx` (`nothing` if the optimizer did
# not report one).
function _fit!(fit::OnlineFit, x, ∇logpx)
    (; lbfgs_state, logp, ∇logp, rng, elbo_rng, ndraws_elbo, elbo_estimates) = fit
    if ∇logpx === nothing
        ∇logp === nothing && throw(
            ArgumentError(
                "The optimizer did not report the gradient at the current point, and " *
                "the optimization problem has no gradient function, so the online fit " *
                "cannot proceed. Provide an `OptimizationFunction` with a `grad`, use an " *
                "optimizer that reports gradients, or use `save_trace=true`.",
            ),
        )
        ∇logpx = ∇logp(x)
    end

    if !fit.started
        # the initial point: fit it, but like `maximize_elbo`, estimate no ELBO for it
        lbfgs_state.θ = x
        lbfgs_state.∇logpθ = ∇logpx
        fit.fit_distribution = fit_mvnormal(lbfgs_state)
        fit.started = true
        return fit
    end

    update_inverse_hessian!(lbfgs_state, x, ∇logpx)
    dist = fit_mvnormal(lbfgs_state)
    Random.seed!(elbo_rng, rand(rng, UInt64))
    estimate = elbo_and_samples(elbo_rng, logp, dist, ndraws_elbo)

    iteration = length(elbo_estimates) + 1
    best = fit.best = _findmax_skipnan_step(fit.best, estimate.value, iteration)
    if last(best) == iteration
        # this fit is now the selected one: keep its draws, and drop the previous winner's
        previous = fit.fit_iteration
        if previous > 0
            elbo_estimates[previous] = _without_draws(elbo_estimates[previous])
        end
        push!(elbo_estimates, estimate)
        fit.fit_distribution = dist
        fit.fit_iteration = iteration
    else
        push!(elbo_estimates, _without_draws(estimate))
    end
    return fit
end

struct OptimizationTrace{P,L}
    points::P
    log_densities::L
    gradients::P
end

# the log densities are always recorded; with an online fit, the points and gradients are not
Base.length(trace::OptimizationTrace) = length(trace.log_densities)

function Base.show(io::IO, trace::OptimizationTrace)
    print(io, "OptimizationTrace with $(length(trace) - 1) iterations")
    return nothing
end
