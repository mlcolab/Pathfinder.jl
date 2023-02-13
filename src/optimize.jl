"""
$(TYPEDEF)

A callable that wraps a LogDensityProblem to be an `fgh!` callable for Optim.jl.

See the [Optim.jl docs](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations)
for details.

# Fields

$(TYPEDFIELDS)
"""
struct OptimJLFunction{P}
    "An object that implements the LogDensityProblem interface"
    prob::P
end

# avoid repeated computation by computing the highest order derivative required and not
# recomputing the lower order ones
function (fun::OptimJLFunction)(F, G, H, x)
    prob = fun.prob
    if H !== nothing
        lp, glp, Hlp = LogDensityProblems.logdensity_gradient_and_hessian(prob, x)
        @. H = -Hlp
        if G !== nothing
            @. G = -glp
        end
        F === nothing || return -lp
    elseif G !== nothing
        lp, glp = LogDensityProblems.logdensity_and_gradient(prob, x)
        @. G = -glp
        F === nothing || return -lp
    elseif F !== nothing
        return -LogDensityProblems.logdensity(prob, x)
    end
    return nothing
end

"""
$(TYPEDEF)

A utility object to mimic a `SciMLBase.OptimizationProblem` for use with Optim.jl.

# Fields

$(TYPEDFIELDS)
"""
struct OptimJLProblem{F<:OptimJLFunction,U<:AbstractVector{<:Real}}
    "An optimization function."
    fun::F
    "Initial point"
    u0::U
end

function _defines_gradient(fun::SciMLBase.OptimizationFunction)
    return fun.grad !== nothing && !(fun.grad isa Bool)
end
_defines_gradient(prob::SciMLBase.OptimizationProblem) = _defines_gradient(prob.f)
_defines_gradient(::Any) = true

"""
$(SIGNATURES)

Construct a log-density function with signature `x -> Real` from an optimization function.
"""
get_logp(fun::SciMLBase.OptimizationFunction) = Base.Fix2((-) ∘ fun.f, nothing)
get_logp(fun::OptimJLFunction) = Base.Fix1(LogDensityProblems.logdensity, fun.prob)

"""
$(SIGNATURES)

Construct a log-density function with signature `x -> Real` from an optimization problem.
"""
get_logp(prob::SciMLBase.OptimizationProblem) = get_logp(prob.f)
get_logp(prob::OptimJLProblem) = get_logp(prob.fun)

"""
$(SIGNATURES)

Build an optimization function from the LogDensityProblem `ℓ`.

The type of the returned object is determined by `optimizer`, either an
[`OptimJLFunction`](@ref) or a `SciMLBase.OptimizationFunction`.
"""
build_optim_function(ℓ, optimizer) = _build_sciml_optim_function(ℓ)
function build_optim_function(
    ℓ, ::Union{Optim.FirstOrderOptimizer,Optim.SecondOrderOptimizer}
)
    return OptimJLFunction(ℓ)
end

function _build_sciml_optim_function(ℓ)
    f(x, p) = -LogDensityProblems.logdensity(ℓ, x)
    function grad(res, x, p)
        _, ∇fx = LogDensityProblems.logdensity_and_gradient(ℓ, x)
        @. res = -∇fx
        return res
    end
    function hess(res, x, p)
        _, _, H = LogDensityProblems.logdensity_gradient_and_hessian(ℓ, x)
        @. res = -H
        return res
    end
    return SciMLBase.OptimizationFunction{true}(f; grad, hess)
end

"""
$(SIGNATURES)

Build an optimization problem from the LogDensityProblem `ℓ` and initial point `x₀`.

The type of the returned object is determined by `optimizer`, either an
[`OptimJLProbkem`](@ref) or a `SciMLBase.OptimizationProblem`.
"""
function build_optim_problem(optim_fun, x₀)
    return SciMLBase.OptimizationProblem(optim_fun, x₀, nothing)
end
function build_optim_problem(optim_fun::OptimJLFunction, x₀)
    return OptimJLProblem(optim_fun, x₀)
end

"""
$(SIGNATURES)

# Returns

- `sol`: The optimization solution, either a `SciMLBase.OptimizationSolution` or a
    `Optim.MultivariateOptimizationResults`
- `trace::OptimizationTrace`: the optimization trace, where the first point is the initial
    one.
"""
function optimize_with_trace(
    prob::SciMLBase.OptimizationProblem,
    optimizer;
    reporter=report_progress,
    try_id=1,
    progress_id=nothing,
    maxiters=1_000,
    callback=nothing,
    fail_on_nonfinite=true,
    kwargs...,
)
    u0 = prob.u0
    fun = prob.f

    function grad(x)
        SciMLBase.isinplace(fun) || return fun.grad(x, nothing)
        res = similar(x)
        fun.grad(res, x, nothing)
        return res
    end

    # allocate containers for the trace of x and f(x)
    xs = typeof(u0)[]
    fxs = typeof(fun.f(u0, nothing))[]
    ∇fxs = typeof(u0)[]
    trace = OptimizationTrace(xs, fxs, ∇fxs)

    _callback = CallbackSequence(
        callback,
        ProgressCallback(; reporter, progress_id, try_id, maxiters, iter_id=0),
        CheckFiniteValueCallback(fail_on_nonfinite),
        FillTraceCallback(grad, trace),
    )
    sol = Optimization.solve(prob, optimizer; callback=_callback, maxiters, kwargs...)
    return sol, trace
end
function optimize_with_trace(
    prob::OptimJLProblem,
    optimizer::Union{Optim.FirstOrderOptimizer,Optim.SecondOrderOptimizer};
    reporter=report_progress,
    try_id=1,
    progress_id=nothing,
    maxiters=1_000,
    callback=nothing,
    fail_on_nonfinite=true,
    kwargs...,
)
    _callback = OptimJLCallbackAdaptor(
        CallbackSequence(
            callback,
            ProgressCallback(; reporter, progress_id, try_id, maxiters, iter_id=0),
            CheckFiniteValueCallback(fail_on_nonfinite),
        ),
    )
    options = Optim.Options(;
        callback=_callback,
        store_trace=true,
        extended_trace=true,
        iterations=maxiters,
        kwargs...,
    )
    result = Optim.optimize(Optim.only_fgh!(prob.fun), prob.u0, optimizer, options)

    u0 = prob.u0
    xtrace = Optim.x_trace(result)
    iterations = min(length(xtrace) - 1, Optim.iterations(result))

    # containers for the trace of x and ∇f(x)
    xs = Vector{typeof(u0)}(undef, iterations + 1)
    ∇fxs = Vector{typeof(u0)}(undef, iterations + 1)

    copyto!(xs, xtrace)
    fxs = -Optim.f_trace(result)
    map!(tr -> -tr.metadata["g(x)"], ∇fxs, Optim.trace(result))
    return result, OptimizationTrace(xs, fxs, ∇fxs)
end
