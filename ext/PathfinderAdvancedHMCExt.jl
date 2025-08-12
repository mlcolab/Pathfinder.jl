module PathfinderAdvancedHMCExt

using AdvancedHMC: AdvancedHMC
using Pathfinder: Pathfinder
using LinearAlgebra: Diagonal, diag
using PDMats: PDMats
using Random: Random

"""
    RankUpdateEuclideanMetric{T,M} <: AdvancedHMC.AbstractMetric

A Gaussian Euclidean metric (mass matrix) whose inverse is constructed by
rank-updates.

To construct this metric, call `AdvancedHMC.AbstractMetric` with a
[`Pathfinder.WoodburyPDMat`](@ref) as the argument.

# Example

```jldoctest
julia> using LinearAlgebra, Pathfinder, AdvancedHMC

julia> W = Pathfinder.WoodburyPDMat(Diagonal([0.1, 0.2]), [0.7 0.2]', Diagonal([0.3]));

julia> AdvancedHMC.AbstractMetric(W)
RankUpdateEuclideanMetric(diag=[0.247, 0.21200000000000002])
```

See also: The AdvancedHMC [metric](@extref AdvancedHMC hamiltonian_mm) documentation.
"""
RankUpdateEuclideanMetric

struct RankUpdateEuclideanMetric{T,M<:Pathfinder.WoodburyPDMat{T,<:Diagonal{T}}} <:
       AdvancedHMC.AbstractMetric
    M⁻¹::M
end

Base.@deprecate(
    Pathfinder.RankUpdateEuclideanMetric(M⁻¹::Pathfinder.WoodburyPDMat),
    AdvancedHMC.AbstractMetric(M⁻¹),
    false,
)

AdvancedHMC.AbstractMetric(M⁻¹::Pathfinder.WoodburyPDMat) = RankUpdateEuclideanMetric(M⁻¹)

AdvancedHMC.renew(::RankUpdateEuclideanMetric, M⁻¹) = RankUpdateEuclideanMetric(M⁻¹)

Base.size(metric::RankUpdateEuclideanMetric, dim...) = size(metric.M⁻¹.A.diag, dim...)

Base.eltype(metric::RankUpdateEuclideanMetric) = eltype(metric.M⁻¹)

function Base.show(io::IO, metric::RankUpdateEuclideanMetric)
    print(io, "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))")
    return nothing
end

function Base.rand(
    rng::Random.AbstractRNG,
    metric::RankUpdateEuclideanMetric{T},
    ::AdvancedHMC.GaussianKinetic,
) where {T}
    M⁻¹ = metric.M⁻¹
    r = Random.randn(rng, T, size(metric)...)
    PDMats.invunwhiten!(r, M⁻¹, r)
    return r
end

function AdvancedHMC.neg_energy(
    h::AdvancedHMC.Hamiltonian{<:RankUpdateEuclideanMetric,<:AdvancedHMC.GaussianKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVecOrMat}
    return -PDMats.quad(h.metric.M⁻¹, r) / 2
end

function AdvancedHMC.∂H∂r(
    h::AdvancedHMC.Hamiltonian{<:RankUpdateEuclideanMetric,<:AdvancedHMC.GaussianKinetic},
    r::AbstractVecOrMat,
)
    return h.metric.M⁻¹ * r
end

end  # module
