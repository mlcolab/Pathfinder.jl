# WoodburyMatrices.SymWoodbury does not work with Distributions.MvNormal, and
# PDMatsExtras.WoodburyPDMat does not support non-diagonal `D`, so we here generalize
# PDMatsExtras.WoodburyPDMat

"""
    WoodburyPDMat <: PDMats.AbstractPDMat

Lazily represents a real positive definite (PD) matrix as an update to a full-rank PD matrix.

    WoodburyPDMat(A, B, D)

Constructs the ``n \\times n`` PD matrix
```math
W = A + B D B^\\mathrm{T},
```
where ``A`` is an ``n \\times n`` full rank positive definite matrix, ``D`` is an
``m \\times m`` symmetric matrix, and ``B`` is an ``n \\times m`` matrix. Note that ``B``
and ``D`` must be chosen such that ``W`` is positive definite; this is only implicitly
checked.

Overloads for `WoodburyPDMat` make extensive use of the following decomposition.
Let ``L_A L_A^\\mathrm{T} = A`` be the Cholesky decomposition of ``A``, and
let ``Q R = L_A^{-1} B`` be a thin QR decomposition. Define ``C = I + RDR^\\mathrm{T}``,
with the Cholesky decomposition ``L_C L_C^\\mathrm{T} = C``. Then, ``W = T T^\\mathrm{T}``,
where
```math
T = L_A Q \\begin{pmatrix} L_C & 0 \\\\ 0 & I \\end{pmatrix}.
```

The positive definite requirement is equivalent to the requirement that both ``A`` and
``C`` are positive definite.

For a derivation of this decomposition for the special case of diagonal ``A``, see
appendix A of [^Zhang2021].

[^Zhang2021]: Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).
              Pathfinder: Parallel quasi-Newton variational inference.
              arXiv: [2108.03782](https://arxiv.org/abs/2108.03782) [stat.ML]
"""
struct WoodburyPDMat{
    T<:Real,
    TA<:AbstractMatrix{T},
    TB<:AbstractMatrix{T},
    TD<:AbstractMatrix{T},
    TUA<:Union{Diagonal{T},UpperTriangular{T}},
    TQ<:AbstractMatrix{T},
    TUC<:Union{Diagonal{T},UpperTriangular{T}},
} <: PDMats.AbstractPDMat{T}
    A::TA
    B::TB
    D::TD
    UA::TUA
    Q::TQ
    UC::TUC
end

function WoodburyPDMat(
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    cholA = cholesky(A)
    UA = cholA.U
    Q, R = qr(UA' \ B)
    cholC = cholesky(Symmetric(muladd(R, D * R', I)))
    return WoodburyPDMat(A, B, D, UA, Q, cholC.U)
end
function WoodburyPDMat(A, B, D)
    T = Base.promote_eltype(A, B, D)
    return WoodburyPDMat(
        convert(AbstractMatrix{T}, A),
        convert(AbstractMatrix{T}, B),
        convert(AbstractMatrix{T}, D),
    )
end

Base.Matrix(W::WoodburyPDMat) = Matrix(Symmetric(muladd(W.B, W.D * W.B', W.A)))

function Base.AbstractMatrix{T}(W::WoodburyPDMat) where {T}
    return WoodburyPDMat(
        map(k -> convert(AbstractMatrix{T}, getfield(W, k)), fieldnames(typeof(W)))...
    )
end

Base.getindex(W::WoodburyPDMat, inds...) = getindex(Matrix(W), inds...)

Base.adjoint(W::WoodburyPDMat) = W

Base.transpose(W::WoodburyPDMat) = W

function Base.inv(W::WoodburyPDMat)
    invUA = inv(W.UA)
    Anew = invUA * invUA'
    Bnew = invUA * Matrix(W.Q)
    invUC = inv(W.UC)
    Dnew = muladd(invUC, invUC', -I)
    return WoodburyPDMat(Anew, Bnew, Dnew)
end

LinearAlgebra.det(W::WoodburyPDMat) = exp(logdet(W))
LinearAlgebra.logdet(W::WoodburyPDMat) = 2 * (logdet(W.UA) + logdet(W.UC))
function LinearAlgebra.logabsdet(W::WoodburyPDMat)
    l = logdet(W)
    return (l, one(l))
end

function LinearAlgebra.diag(W::WoodburyPDMat)
    D = W.D isa Diagonal ? W.D : Symmetric(W.D)
    return diag(W.A) + map(b -> dot(b, D, b), eachrow(W.B))
end

function LinearAlgebra.lmul!(W::WoodburyPDMat, x::StridedVecOrMat)
    UA = W.UA
    UC = W.UC
    Q = W.Q
    k = minimum(size(W.B))
    lmul!(UA, x)
    lmul!(Q', x)
    x1 = x isa AbstractVector ? view(x, 1:k) : view(x, 1:k, :)
    lmul!(UC, x1)
    lmul!(UC', x1)
    lmul!(Q, x)
    lmul!(UA', x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end
function LinearAlgebra.mul!(y::AbstractMatrix, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end

function Base.:*(W::WoodburyPDMat, c::Real)
    c > 0 || return Matrix(W) * c
    return WoodburyPDMat(W.A * c, W.B * one(c), W.D * c)
end

PDMats.dim(W::WoodburyPDMat) = size(W.A, 1)

function PDMats.invquad(W::WoodburyPDMat{<:Real}, x::AbstractVector{<:Real})
    v = W.Q' * (W.UA' \ x)
    n, m = size(W.B)
    k = min(m, n)
    return @views sum(abs2, W.UC' \ v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.unwhiten!(r::StridedVecOrMat, W::WoodburyPDMat, x::StridedVecOrMat)
    k = minimum(size(W.B))
    copyto!(r, x)
    @views lmul!(W.UC', r[1:k])
    lmul!(W.Q, r)
    lmul!(W.UA', r)
    return r
end
