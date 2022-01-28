function build_optim_function(f, ∇f)
    fun(x, _) = -f(x)
    function grad!(n∇fx, x, _...)
        n∇fx .= .-∇f(x)
        return n∇fx
    end
    # TODO: use AbstractDifferentiation to provide Hessian, etc in case needed
    return GalacticOptim.OptimizationFunction(
        fun, GalacticOptim.AutoForwardDiff(); grad=grad!
    )
end

function build_optim_problem(optim_fun, x₀; kwargs...)
    return GalacticOptim.OptimizationProblem(optim_fun, x₀; kwargs...)
end

function optimize_with_trace(prob, optimizer; kwargs...)
    u0 = prob.u0
    fun = prob.f
    grad! = fun.grad
    function ∇f(x)
        ∇fx = similar(x)
        grad!(∇fx, x)
        rmul!(∇fx, -1)
        return ∇fx
    end
    # caches for the trace of x, f(x), and ∇f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, Any))[]
    ∇fxs = typeof(similar(u0))[]
    function callback(x, nfx, args...)
        # NOTE: GalacticOptim doesn't have an interface for accessing the gradient trace,
        # so we need to recompute it ourselves
        # see https://github.com/SciML/GalacticOptim.jl/issues/149
        ∇fx = ∇f(x)
        # terminate if optimization encounters NaNs
        (isnan(nfx) || any(isnan, x) || any(isnan, ∇fx)) && return true
        # some backends mutate x, so we must copy it
        push!(xs, copy(x))
        push!(fxs, -nfx)
        push!(∇fxs, ∇fx)
        return false
    end
    GalacticOptim.solve(prob, optimizer; cb=callback)
    return xs, fxs, ∇fxs
end
