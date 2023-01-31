using .AdvancedHMC: AdvancedHMC
using Random

"""
    RankUpdateEuclideanMetric{T,M} <: AdvancedHMC.AbstractMetric

A Gaussian Euclidean metric whose inverse is constructed by rank-updates.

# Constructors

    RankUpdateEuclideanMetric(n::Int)
    RankUpdateEuclideanMetric(M⁻¹::Pathfinder.WoodburyPDMat)

Construct a Gaussian Euclidean metric of size `(n, n)` with inverse of `M⁻¹`.

# Example

```jldoctest
julia> using LinearAlgebra, Pathfinder, AdvancedHMC

julia> Pathfinder.RankUpdateEuclideanMetric(3)
RankUpdateEuclideanMetric(diag=[1.0, 1.0, 1.0])

julia> W = Pathfinder.WoodburyPDMat(Diagonal([0.1, 0.2]), [0.7 0.2]', Diagonal([0.3]))
2×2 Pathfinder.WoodburyPDMat{Float64, Diagonal{Float64, Vector{Float64}}, Adjoint{Float64, Matrix{Float64}}, Diagonal{Float64, Vector{Float64}}, Diagonal{Float64, Vector{Float64}}, QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}, UpperTriangular{Float64, Matrix{Float64}}}:
 0.247  0.042
 0.042  0.212

julia> Pathfinder.RankUpdateEuclideanMetric(W)
RankUpdateEuclideanMetric(diag=[0.247, 0.21200000000000002])
```
"""
RankUpdateEuclideanMetric

struct RankUpdateEuclideanMetric{T,M<:WoodburyPDMat{T,<:Diagonal{T}}} <:
       AdvancedHMC.AbstractMetric
    M⁻¹::M
end

function RankUpdateEuclideanMetric(n::Int)
    M⁻¹ = WoodburyPDMat(Diagonal(ones(n)), zeros(n, 0), zeros(0, 0))
    return RankUpdateEuclideanMetric(M⁻¹)
end
function RankUpdateEuclideanMetric(::Type{T}, D::Int) where {T}
    return RankUpdateEuclideanMetric(
        WoodburyPDMat(Diagonal(ones(T, D)), Matrix{T}(undef, D, 0), Matrix{T}(undef, 0, 0))
    )
end
function RankUpdateEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T}
    return RankUpdateEuclideanMetric(T, first(sz))
end
RankUpdateEuclideanMetric(sz::Tuple{Int}) = RankUpdateEuclideanMetric(Float64, sz)

AdvancedHMC.renew(::RankUpdateEuclideanMetric, M⁻¹) = RankUpdateEuclideanMetric(M⁻¹)

Base.size(metric::RankUpdateEuclideanMetric, dim...) = size(metric.M⁻¹, dim...)

function Base.show(io::IO, metric::RankUpdateEuclideanMetric)
    print(io, "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))")
    return nothing
end

function Base.rand(
    rng::AbstractRNG, metric::RankUpdateEuclideanMetric{T}, ::AdvancedHMC.GaussianKinetic
) where {T}
    M⁻¹ = metric.M⁻¹
    r = randn(rng, T, size(metric)...)
    invunwhiten!(r, M⁻¹, r)
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
