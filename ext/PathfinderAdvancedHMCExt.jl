module PathfinderAdvancedHMCExt

using AdvancedHMC: AdvancedHMC
using LinearAlgebra: Diagonal
using Pathfinder: Pathfinder

@deprecate Pathfinder.RankUpdateEuclideanMetric(
    W::Pathfinder.WoodburyPDMat{T,<:Diagonal{T}} where {T}
) AdvancedHMC.RankUpdateEuclideanMetric(W) false

"""
    AdvancedHMC.RankUpdateEuclideanMetric(W::WoodburyPDMat)

Construct an `AdvancedHMC.RankUpdateEuclideanMetric` from a `WoodburyPDMat`,
reusing its precomputed factorization.
"""
function AdvancedHMC.RankUpdateEuclideanMetric(
    W::Pathfinder.WoodburyPDMat{T,<:Diagonal{T}}
) where {T}
    return AdvancedHMC.RankUpdateEuclideanMetric(
        W.A, W.B, W.D, AdvancedHMC.WoodburyFactorization(W.F.U, W.F.Q, W.F.V)
    )
end

end  # module
