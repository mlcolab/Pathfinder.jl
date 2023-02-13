# utility macro to annotate a field as const only if supported
@eval macro $(Symbol("const"))(x)
    if VERSION ≥ v"1.8"
        Expr(:const, esc(x))
    else
        esc(x)
    end
end

# callbacks for Optimization.jl

"""
$(TYPEDEF)

Abstract type for Optimization.jl callbacks.

A callback should be a callable with the signature

    (x, f(x), args...) -> Bool

where `x` is the parameter being optimized, and `f` is the objective function. A return
value of `true` signals that optimization should stop.

See the [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/solve/) for
more information.
"""
abstract type AbstractOptimizationCallback end

"""
$(TYPEDEF)

A sequence of Optimization.jl callbacks to be executed in order.
"""
struct CallbackSequence{C<:Tuple} <: AbstractOptimizationCallback
    "Tuple of Optimization.jl callbacks to be called in order"
    callbacks::C
end

"""
$(SIGNATURES)

Wrap the sequence `callbacks`.
"""
CallbackSequence(callbacks...) = CallbackSequence(callbacks)

function (callback::CallbackSequence)(args...)
    return mapfoldl(|, callback.callbacks; init=false) do cb
        cb === nothing && return false
        return cb(args...)
    end
end

"""
$(TYPEDEF)

A callback to log progress with a `reporter`

# Fields

$(FIELDS)
"""
Base.@kwdef mutable struct ProgressCallback{R} <: AbstractOptimizationCallback
    "Reporter function, called with signature `report(progress_id, maxiters, try_id, iter_id)`"
    @const reporter::R
    "An identifier for the progress bar."
    @const progress_id::Base.UUID
    "Maximum number of iterations"
    @const maxiters::Int
    "Index of the current try"
    try_id::Int
    "Index of the current iteration"
    iter_id::Int
end

function (cb::ProgressCallback)(args...)
    cb.iter_id += 1
    return false
end
(::ProgressCallback{Nothing})(args...) = false

"""
$(SIGNATURES)

Report progress using ProgressLogging.jl.
"""
function report_progress(progress_id::Base.UUID, maxiters::Int, try_id::Int, iter_id::Int)
    Base.@logmsg ProgressLogging.ProgressLevel "Optimizing (try $(try_id))" progress =
        iter_id / maxiters _id = progress_id
    return nothing
end

"""
$(TYPEDEF)

A callback that signals termination if the objective value is non-finite and `fail=true`.
"""
struct CheckFiniteValueCallback <: AbstractOptimizationCallback
    "Whether to raise an error if the objective function is non-finite"
    fail::Bool
end

function (cb::CheckFiniteValueCallback)(x, fx, ∇fx, args...)
    return cb.fail && isnan(fx) || fx == -Inf || any(!isfinite, ∇fx)
end

struct FillTraceCallback{G,T<:OptimizationTrace} <: AbstractOptimizationCallback
    "A function to compute the gradient of the objective function"
    grad::G
    "An `Optimization` with empty vectors to be filled."
    trace::T
end

function (cb::FillTraceCallback)(x, fx, args...)
    # NOTE: Optimization doesn't have an interface for accessing the gradient trace,
    # so we need to recompute it ourselves
    # see https://github.com/SciML/Optimization.jl/issues/149
    ∇fx = cb.grad(x)
    rmul!(∇fx, -1)

    trace = cb.trace
    # some backends mutate x, so we must copy it
    push!(trace.points, copy(x))
    push!(trace.log_densities, -fx)
    push!(trace.gradients, ∇fx)
    return false
end

# callbacks for Optim.jl

"""
$(TYPEDEF)

Abstract type for Optim.jl callbacks.

A callback should be a callable with the signature

    (states::Vector{<:AbstractOptimizerState}) -> Bool

where `x` is the parameter being optimized, and `f` is the objective function. A return
value of `true` signals that optimization should stop.
"""
abstract type AbstractOptimJLCallback end

"""
$(TYPEDEF)

Adaptor for an Optimization.jl callback to be an Optim.jl callback.
"""
struct OptimJLCallbackAdaptor{C} <: AbstractOptimJLCallback
    "An Optimization.jl callback to be called."
    callback::C
end

function (cb::OptimJLCallbackAdaptor)(states)
    state = states[end]
    md = state.metadata
    x = md["x"]
    fx = state.value
    return cb.callback(x, fx, md["g(x)"])
end
(::OptimJLCallbackAdaptor{Nothing})(states) = false
