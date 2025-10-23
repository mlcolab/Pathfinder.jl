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
    kwargs...,
)
    u0 = prob.u0
    fun = prob.f
    # caches for the trace of x and f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = Union{Nothing,typeof(u0)}[]
    _callback = OptimizationCallback(
        xs, fxs, ∇fxs, progress_name, progress_id, maxiters, callback, fail_on_nonfinite
    )
    sol = Optimization.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)

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

struct OptimizationCallback{X,FX,∇FX,ID,CB}
    xs::X
    fxs::FX
    ∇fxs::∇FX
    progress_name::String
    progress_id::ID
    maxiters::Int
    callback::CB
    fail_on_nonfinite::Bool
end

function (cb::OptimizationCallback)(state::OptimizationState, args...)
    (; xs, fxs, ∇fxs, progress_name, progress_id, maxiters, callback, fail_on_nonfinite) =
        cb
    ret = callback !== nothing && callback(state, args...)
    iteration = state.iter
    Base.@logmsg ProgressLogging.ProgressLevel progress_name progress = iteration / maxiters _id =
        progress_id

    x = copy(state.u)
    fx = -state.objective
    ∇fx = state.grad === nothing ? nothing : -state.grad

    # some backends mutate x, so we must copy it
    push!(xs, x)
    push!(fxs, fx)
    push!(∇fxs, ∇fx)

    if fail_on_nonfinite && !ret
        ret = (isnan(fx) || fx == Inf || (∇fx !== nothing && any(!isfinite, ∇fx)))::Bool
    end

    return ret
end

struct OptimizationTrace{P,L}
    points::P
    log_densities::L
    gradients::P
end

Base.length(trace::OptimizationTrace) = length(trace.points)

function Base.show(io::IO, trace::OptimizationTrace)
    print(io, "OptimizationTrace with $(length(trace) - 1) iterations")
    return nothing
end
