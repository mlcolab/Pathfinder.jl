# WoodburyMatrices.SymWoodbury does not work with Distributions.MvNormal, and
# PDMatsExtras.WoodburyPDMat does not support non-diagonal `D`, so we here generalize
# PDMatsExtras.WoodburyPDMat

"""
    WoodburyPDMat(A::AbstractMatrix{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T<:Real}

Lazily represents symmetric matrices constructed from a low-rank update, that is
```math
W = A + B D B',
```
where ``A`` is a full rank positive definite matrix, ``D`` is a rank-``k`` symmetric matrix,
and ``B`` is a matrix of compatible size. Note that ``B`` and ``D`` must be chosen such that
``W`` is positive definite; this is only implicitly checked.

Overloads for `WoodburyPDMat` make extensive use of the following decomposition.
Let ``L_A L_A^\\mathrm{T} = A`` be the Cholesky decomposition of ``A``, and
let ``Q R = L_A^{-1} B`` be a thin QR decomposition. Define ``C = I + RDR^\\mathrm{T}``.
Then, ``W = T T^\\mathrm{T}``, where
```math
T = L_A Q \\begin{pmatrix} L_C & 0 \\\\ 0 & I \\end{pmatrix}.
```

For a derivation of this decomposition for the special case of diagonal ``A``, see
appendix A of [^Zhang2021].

[^Zhang2021] Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).
             Pathfinder: Parallel quasi-Newton variational inference.
             arXiv: [2108.03782](https://arxiv.org/abs/2108.03782) [stat.ML]
"""
struct WoodburyPDMat{
    T<:Real,
    TA<:AbstractMatrix{T},
    TB<:AbstractMatrix{T},
    TD<:AbstractMatrix{T},
    TcholA<:Cholesky{T},
    TQ<:AbstractMatrix{T},
    TcholC<:Cholesky{T},
} <: PDMats.AbstractPDMat{T}
    A::TA
    B::TB
    D::TD
    cholA::TcholA
    Q::TQ
    cholC::TcholC
end

function WoodburyPDMat(A, B, D)
    cholA = cholesky(A)
    QR = qr(cholA.U' \ B)
    R = UpperTriangular(QR.R)
    cholC = cholesky(Symmetric(muladd(R, D * R', I)))
    return WoodburyPDMat(A, B, D, cholA, QR.Q, cholC)
end

PDMats.dim(W::WoodburyPDMat) = size(W.A, 1)

Base.Matrix(W::WoodburyPDMat) = Matrix(Symmetric(muladd(W.B, W.D * W.B', W.A)))

Base.getindex(W::WoodburyPDMat, inds...) = getindex(Matrix(W), inds...)

function Base.inv(W::WoodburyPDMat)
    invLA = inv(W.cholA.U')
    A = invLA' * invLA
    B = invLA' * Matrix(W.Q)
    D = inv(W.cholC) - I
    return WoodburyPDMat(A, B, D)
end

LinearAlgebra.det(W::WoodburyPDMat) = exp(logdet(W))
LinearAlgebra.logdet(W::WoodburyPDMat) = logdet(W.cholA) + logdet(W.cholC)
function LinearAlgebra.logabsdet(W::WoodburyPDMat)
    l = logdet(W)
    return (l, one(l))
end

function LinearAlgebra.diag(W::WoodburyPDMat)
    D = Symmetric(W.D)
    return diag(W.A) + map(b -> dot(b, D, b), eachrow(W.B))
end

function PDMats.invquad(W::WoodburyPDMat{<:Real}, x::AbstractVector{<:Real})
    v = W.Q' * (W.cholA.U' \ x)
    n, k = size(W.B)
    return @views sum(abs2, W.cholC.U' \ v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.unwhiten!(r::DenseVecOrMat, W::WoodburyPDMat, x::DenseVecOrMat)
    k = size(W.B, 2)
    copyto!(r, x)
    @views lmul!(W.cholC.U', r[1:k])
    lmul!(W.Q, r)
    lmul!(W.cholA.U', r)
    return r
end

Base.adjoint(W::WoodburyPDMat) = W

Base.transpose(W::WoodburyPDMat) = W

function LinearAlgebra.lmul!(W::WoodburyPDMat, x::StridedVecOrMat)
    LA = W.cholA.U'
    LC = W.cholC.U'
    Q = W.Q
    k = size(W.B, 2)
    lmul!(LA', x)
    lmul!(Q', x)
    x1 = x isa AbstractVector ? view(x, 1:k) : view(x, 1:k, :)
    lmul!(LC', x1)
    lmul!(LC, x1)
    lmul!(Q, x)
    lmul!(LA, x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end
function LinearAlgebra.mul!(y::AbstractMatrix, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end

Base.:*(a::WoodburyPDMat, c::Real) = WoodburyPDMat(a.A * c, a.B, a.D * c)
