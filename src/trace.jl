"""
    $(TYPEDEF)

A container for the trajectory of points and values computed during optimization.

# Fields

$(FIELDS)
"""
struct OptimizationTrace{P,L}
    "Points in the optimization trajectory"
    points::P
    "Log-density (negative objective function) values at `points`"
    log_densities::L
    "Gradient of the log-density values at `points`"
    gradients::P
end

Base.length(trace::OptimizationTrace) = length(trace.points)

function Base.show(io::IO, trace::OptimizationTrace)
    print(io, "OptimizationTrace with $(length(trace) - 1) iterations")
    return nothing
end
