function build_optim_function(
    log_density_problem,
    adtype::ADTypes.AbstractADType,
    ::LogDensityProblems.LogDensityOrder,
)
    function grad(res, x, _...)
        _, ∇fx = LogDensityProblems.logdensity_and_gradient(log_density_problem, x)
        @. res = -∇fx
        return res
    end
    function hess(res, x, _...)
        _, _, H = LogDensityProblems.logdensity_gradient_and_hessian(log_density_problem, x)
        @. res = -H
        return res
    end
    return build_optim_function(
        Base.Fix1(LogDensityProblems.logdensity, log_density_problem), adtype; grad, hess
    )
end
function build_optim_function(
    log_density_problem,
    adtype::ADTypes.AbstractADType,
    ::LogDensityProblems.LogDensityOrder{0};
    kwargs...,
)
    return build_optim_function(
        Base.Fix1(LogDensityProblems.logdensity, log_density_problem), adtype; kwargs...
    )
end
function build_optim_function(log_density_fun, adtype::ADTypes.AbstractADType; kwargs...)
    f(x, _...) = -log_density_fun(x)
    return SciMLBase.OptimizationFunction(f, adtype; kwargs...)
end

build_optim_problem(optim_fun, x₀) = SciMLBase.OptimizationProblem(optim_fun, x₀, nothing)

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
    function ∇f(x)
        SciMLBase.isinplace(fun) || return fun.grad(x, nothing)
        res = similar(x)
        fun.grad(res, x, nothing)
        rmul!(res, -1)
        return res
    end
    # caches for the trace of x and f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = typeof(u0)[]
    _callback = OptimizationCallback(
        xs, fxs, ∇fxs, ∇f, progress_name, progress_id, maxiters, callback, fail_on_nonfinite
    )
    sol = Optimization.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)
    return sol, OptimizationTrace(xs, fxs, ∇fxs)
end

struct OptimizationCallback{X,FX,∇FX,∇F,ID,CB}
    xs::X
    fxs::FX
    ∇fxs::∇FX
    ∇f::∇F
    progress_name::String
    progress_id::ID
    maxiters::Int
    callback::CB
    fail_on_nonfinite::Bool
end

@static if isdefined(Optimization, :OptimizationState)
    # Optimization v3.21.0 and later
    function (cb::OptimizationCallback)(state::Optimization.OptimizationState, args...)
        @unpack (
            xs,
            fxs,
            ∇fxs,
            ∇f,
            progress_name,
            progress_id,
            maxiters,
            callback,
            fail_on_nonfinite,
        ) = cb
        ret = callback !== nothing && callback(state, args...)
        iteration = state.iter
        Base.@logmsg ProgressLogging.ProgressLevel progress_name progress =
            iteration / maxiters _id = progress_id

        x = copy(state.u)
        fx = -state.objective
        ∇fx = state.grad === nothing ? ∇f(x) : -state.grad

        # some backends mutate x, so we must copy it
        push!(xs, x)
        push!(fxs, fx)
        push!(∇fxs, ∇fx)

        if fail_on_nonfinite && !ret
            ret = (isnan(fx) || fx == Inf || any(!isfinite, ∇fx))::Bool
        end

        return ret
    end
else
    # Optimization v3.20.X and earlier
    function (cb::OptimizationCallback)(x, nfx, args...)
        @unpack (
            xs,
            fxs,
            ∇fxs,
            ∇f,
            progress_name,
            progress_id,
            maxiters,
            callback,
            fail_on_nonfinite,
        ) = cb
        ret = callback !== nothing && callback(x, nfx, args...)
        iteration = length(cb.xs)
        Base.@logmsg ProgressLogging.ProgressLevel progress_name progress =
            iteration / maxiters _id = progress_id

        # some backends mutate x, so we must copy it
        push!(xs, copy(x))
        push!(fxs, -nfx)
        # NOTE: Optimization doesn't have an interface for accessing the gradient trace,
        # so we need to recompute it ourselves
        # see https://github.com/SciML/Optimization.jl/issues/149
        ∇fx = ∇f(x)
        push!(∇fxs, ∇fx)

        if fail_on_nonfinite && !ret
            ret = (isnan(nfx) || nfx == -Inf || any(!isfinite, ∇fx))::Bool
        end

        return ret
    end
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
