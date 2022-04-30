function build_optim_function(f; ad_backend=AD.ForwardDiffBackend())
    ∇f(x) = only(AD.gradient(ad_backend, f, x))
    return build_optim_function(f, ∇f; ad_backend)
end
function build_optim_function(f, ∇f; ad_backend=AD.ForwardDiffBackend())
    # because we need explicit access to grad, we generate these ourselves instead of using
    # GalacticOptim's auto-AD feature.
    # TODO: switch to caching API if available, see
    # https://github.com/JuliaDiff/AbstractDifferentiation.jl/issues/41
    function grad(res, x, p...)
        ∇fx = ∇f(x)
        @. res = -∇fx
        return res
    end
    function hess(res, x, p...)
        H = only(AD.hessian(ad_backend, f, x))
        @. res = -H
        return res
    end
    function hv(res, x, v, p...)
        Hv = only(AD.lazy_hessian(ad_backend, f, x) * v)
        @. res = -Hv
        return res
    end
    return GalacticOptim.OptimizationFunction((x, p...) -> -f(x); grad, hess, hv)
end

function build_optim_problem(optim_fun, x₀)
    return GalacticOptim.OptimizationProblem(optim_fun, x₀, nothing)
end

function optimize_with_trace(
    prob,
    optimizer;
    progress_name="Optimizing",
    progress_id=nothing,
    maxiters=1_000,
    cb=nothing,
    kwargs...,
)
    u0 = prob.u0
    fun = prob.f
    grad! = fun.grad
    # caches for the trace of x and f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = typeof(u0)[]
    iteration = 0
    function callback(x, nfx, args...)
        ret = cb !== nothing && cb(x, nfx, args...)

        Base.@logmsg ProgressLogging.ProgressLevel progress_name progress =
            iteration / maxiters _id = progress_id

        iteration += 1
        # some backends mutate x, so we must copy it
        push!(xs, copy(x))
        push!(fxs, -nfx)
        # NOTE: GalacticOptim doesn't have an interface for accessing the gradient trace,
        # so we need to recompute it ourselves
        # see https://github.com/SciML/GalacticOptim.jl/issues/149
        push!(∇fxs, rmul!(grad!(similar(x), x, nothing), -1))
        return ret
    end
    sol = GalacticOptim.solve(prob, optimizer; cb=callback, maxiters, kwargs...)
    return sol, OptimizationTrace(xs, fxs, ∇fxs)
end

function optimize_with_trace(
    prob,
    optimizer::Union{Optim.FirstOrderOptimizer,Optim.SecondOrderOptimizer};
    progress_name="Optimizing",
    progress_id=nothing,
    maxiters=1_000,
    cb=nothing,
    kwargs...,
)
    iteration = 0
    function callback(x, nfx, args...)
        ret = cb !== nothing && cb(x, nfx, args...)
        Base.@logmsg ProgressLogging.ProgressLevel progress_name progress =
            iteration / maxiters _id = progress_id
        iteration += 1
        return ret
    end
    new_kwargs = merge(NamedTuple(kwargs), (; store_trace=true, extended_trace=true))
    sol = GalacticOptim.solve(prob, optimizer; cb=callback, maxiters, new_kwargs...)

    u0 = prob.u0
    xs = Vector{typeof(u0)}(undef, iteration)
    ∇fxs = Vector{typeof(u0)}(undef, iteration)
    result = sol.original
    copyto!(xs, Optim.x_trace(result))
    fxs = -Optim.f_trace(result)
    map!(tr -> -tr.metadata["g(x)"], ∇fxs, Optim.trace(result))

    return sol, OptimizationTrace(xs::Vector{typeof(u0)}, fxs, ∇fxs::Vector{typeof(u0)})
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
