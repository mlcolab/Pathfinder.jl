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

function build_optim_problem(optim_fun, x₀; kwargs...)
    return GalacticOptim.OptimizationProblem(optim_fun, x₀, nothing; kwargs...)
end

function optimize_with_trace(prob, optimizer)
    u0 = prob.u0
    fun = prob.f
    grad! = fun.grad
    function ∇f(x)
        ∇fx = similar(x)
        grad!(∇fx, x, nothing)
        rmul!(∇fx, -1)
        return ∇fx
    end
    # caches for the trace of x, f(x), and ∇f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = typeof(similar(u0))[]
    function callback(x, nfx, args...)
        # NOTE: GalacticOptim doesn't have an interface for accessing the gradient trace,
        # so we need to recompute it ourselves
        # see https://github.com/SciML/GalacticOptim.jl/issues/149
        ∇fx = ∇f(x)
        # some backends mutate x, so we must copy it
        push!(xs, copy(x))
        push!(fxs, -nfx)
        push!(∇fxs, ∇fx)
        return false
    end
    GalacticOptim.solve(prob, optimizer; cb=callback)
    return xs, fxs, ∇fxs
end
