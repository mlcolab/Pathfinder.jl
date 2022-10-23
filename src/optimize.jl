function build_optim_function(f; ad_backend=AD.ForwardDiffBackend())
    ∇f(x) = only(AD.gradient(ad_backend, f, x))
    return build_optim_function(f, ∇f; ad_backend)
end
function build_optim_function(f, ∇f; ad_backend=AD.ForwardDiffBackend())
    # because we need explicit access to grad, we generate these ourselves instead of using
    # Optimization.jl's auto-AD feature.
    # TODO: switch to caching API if available, see
    # https://github.com/JuliaDiff/AbstractDifferentiation.jl/issues/41
    function grad(res, x, p)
        ∇fx = ∇f(x)
        @. res = -∇fx
        return res
    end
    function hess(res, x, p)
        H = only(AD.hessian(ad_backend, f, x))
        @. res = -H
        return res
    end
    function hv(res, x, v, p)
        Hv = only(AD.lazy_hessian(ad_backend, f, x) * v)
        @. res = -Hv
        return res
    end
    return SciMLBase.OptimizationFunction{true}((x, p) -> -f(x); grad, hess, hv)
end

function build_optim_problem(optim_fun, x₀)
    return SciMLBase.OptimizationProblem(optim_fun, x₀, nothing)
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
    _callback = _make_optimization_callback(
        xs, fxs, ∇fxs, ∇f; progress_name, progress_id, maxiters, callback, fail_on_nonfinite
    )
    sol = Optimization.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)
    return sol, OptimizationTrace(xs, fxs, ∇fxs)
end

function _make_optimization_callback(
    xs, fxs, ∇fxs, ∇f; progress_name, progress_id, maxiters, callback, fail_on_nonfinite
)
    return function (x, nfx, args...)
        ret = callback !== nothing && callback(x, nfx, args...)
        iteration = length(xs)
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
