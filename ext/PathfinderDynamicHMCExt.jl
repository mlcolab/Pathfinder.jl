module PathfinderDynamicHMCExt

using DynamicHMC: DynamicHMC
using Pathfinder: Pathfinder
using PDMats: PDMats

function DynamicHMC.GaussianKineticEnergy(M⁻¹::Pathfinder.WoodburyPDMat)
    return DynamicHMC.GaussianKineticEnergy(M⁻¹, inv(Pathfinder.pdfactorize(M⁻¹).R))
end

function DynamicHMC.kinetic_energy(
    κ::DynamicHMC.GaussianKineticEnergy{<:Pathfinder.WoodburyPDMat}, p, q=nothing
)
    return PDMats.quad(κ.M⁻¹, p) / 2
end

end  # module
