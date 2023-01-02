module DynamicHMCExt

using PDMats: PDMats
if isdefined(Base, :get_extension)
    using Pathfinder: Pathfinder
    using DynamicHMC: DynamicHMC
else  # using Requires
    using ..Pathfinder: Pathfinder
    using ..DynamicHMC: DynamicHMC
end

function DynamicHMC.GaussianKineticEnergy(M⁻¹::Pathfinder.WoodburyPDMat)
    return DynamicHMC.GaussianKineticEnergy(M⁻¹, inv(Pathfinder.pdfactorize(M⁻¹).R))
end

function DynamicHMC.kinetic_energy(
    κ::DynamicHMC.GaussianKineticEnergy{<:Pathfinder.WoodburyPDMat}, p, q=nothing
)
    return PDMats.quad(κ.M⁻¹, p) / 2
end

end  # module
