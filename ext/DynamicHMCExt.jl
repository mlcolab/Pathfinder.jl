using .DynamicHMC: DynamicHMC

function DynamicHMC.GaussianKineticEnergy(M⁻¹::WoodburyPDMat)
    return DynamicHMC.GaussianKineticEnergy(M⁻¹, inv(pdfactorize(M⁻¹).R))
end

function DynamicHMC.kinetic_energy(
    κ::DynamicHMC.GaussianKineticEnergy{<:WoodburyPDMat}, p, q=nothing
)
    return PDMats.quad(κ.M⁻¹, p) / 2
end
