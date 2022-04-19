using .DynamicHMC: DynamicHMC

function DynamicHMC.GaussianKineticEnergy(M⁻¹::WoodburyPDMat)
    return DynamicHMC.GaussianKineticEnergy(M⁻¹, WoodburyLeftInvFactor(M⁻¹))
end

function DynamicHMC.kinetic_energy(
    κ::DynamicHMC.GaussianKineticEnergy{<:WoodburyPDMat}, p, q=nothing
)
    return PDMats.quad(κ.M⁻¹, p) / 2
end

# utility object so we can use DynamicHMC.GaussianKineticEnergy
struct WoodburyLeftInvFactor{T<:Real,TW<:PDMats.AbstractPDMat{T}} <: AbstractMatrix{T}
    A::TW
end

Base.size(L::WoodburyLeftInvFactor, dims::Int...) = size(L.A, dims...)

function LinearAlgebra.mul!(
    r::StridedVecOrMat{T}, L::WoodburyLeftInvFactor{T}, x::StridedVecOrMat{T}
) where {T}
    return invunwhiten!(r, L.A, x)
end

function Base.Matrix(L::WoodburyLeftInvFactor)
    W = L.A
    n, m = size(W.B)
    k = min(m, n)
    Lmat = zeros(n, n)
    Lmat[diagind(Lmat)] .= true
    @views ldiv!(W.UC, Lmat[1:k, 1:k])
    lmul!(W.Q, Lmat)
    ldiv!(W.UA, Lmat)
    return Lmat
end

function Base.AbstractMatrix{T}(L::WoodburyLeftInvFactor) where {T}
    return WoodburyLeftInvFactor(AbstractMatrix{T}(L.A))
end

Base.getindex(L::WoodburyLeftInvFactor, inds...) = getindex(Matrix(L), inds...)
