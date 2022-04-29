using .MeasureTheory: MeasureTheory
using .TransformVariables: TransformVariables

function pathfinder(measure::MeasureTheory.AbstractMeasure; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    return pathfinder(logp; dim=TransformVariables.dimension(logp.transform), kwargs...)
end

function multipathfinder(measure::MeasureTheory.AbstractMeasure, ndraws::Int; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    dim = TransformVariables.dimension(logp.transform)
    return multipathfinder(logp, ndraws; dim, kwargs...)
end

struct MeasureTheoryLogDensity{M,T}
    measure::M
    transform::T
end
function MeasureTheoryLogDensity(m::MeasureTheory.AbstractMeasure)
    return MeasureTheoryLogDensity(m, MeasureTheory.as(m))
end

function (logp::MeasureTheoryLogDensity)(z)
    x, logdetJ = TransformVariables.transform_and_logjac(logp.transform, z)
    return MeasureTheory.logdensityof(logp.measure, x) + logdetJ
end
