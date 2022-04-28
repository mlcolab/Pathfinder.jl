using .MeasureTheory: MeasureTheory
using .TransformVariables: TransformVariables

function pathfinder(measure::MeasureTheory.AbstractMeasure; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    return pathfinder(logp; dim=TransformVariables.dimension(trans), kwargs...)
end

function multipathfinder(measure::MeasureTheory.AbstractMeasure, ndraws::Int; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    dim = TransformVariables.dimension(logp.trans)
    return multipathfinder(logp, ndraws; dim, kwargs...)
end

struct MeasureTheoryLogDensity{M,T}
    measure::M
    transform::T
end
MeasureTheoryLogDensity(m::Measure) = MeasureTheoryLogDensity(m, MeasureTheory.xform(m))

function (logp::MeasureTheoryLogDensity)(x)
    x, logdetJ = TransformVariables.transform_and_logjac(trans, z)
    return MeasureTheory.logdensityof(measure, x) + logdetJ
end
