module PathfinderAdvancedHMCExt

using AdvancedHMC: AdvancedHMC
using LinearAlgebra: Diagonal
using Pathfinder: Pathfinder

function AdvancedHMC.RankUpdateEuclideanMetric(
    W::Pathfinder.WoodburyPDMat{T,<:Diagonal{T}}
) where {T}
    return AdvancedHMC.RankUpdateEuclideanMetric(W.A, W.B, W.D)
end

end  # module
