# WoodburyMatrices.SymWoodbury does not work with Distributions.MvNormal, and
# PDMatsExtras.WoodburyPDMat does not support non-diagonal `D`, so we here generalize
# PDMatsExtras.WoodburyPDMat

"""
    WoodburyPDFactorization{T,F} <: Factorization{T}

A "square root" factorization of a positive definite Woodbury matrix.

See [`pdfactorize`](@ref), [`WoodburyPDRightFactor`](@ref), [`WoodburyPDMat`](@ref).
"""
struct WoodburyPDFactorization{
    T<:Real,
    TU<:Union{Diagonal{T},LinearAlgebra.AbstractTriangular{T}},
    TQ, # wide type to support any Q
    TV<:Union{Diagonal{T},LinearAlgebra.AbstractTriangular{T}},
} <: Factorization{T}
    U::TU
    Q::TQ
    V::TV
end

function Base.getproperty(F::WoodburyPDFactorization, s::Symbol)
    if s === :R
        return WoodburyPDRightFactor(getfield(F, :U), getfield(F, :Q), getfield(F, :V))
    elseif s === :L
        return transpose(
            WoodburyPDRightFactor(getfield(F, :U), getfield(F, :Q), getfield(F, :V))
        )
    else
        return getfield(F, s)
    end
end

function Base.propertynames(F::WoodburyPDFactorization, private::Bool=false)
    return (:L, :R, (private ? fieldnames(typeof(F)) : ())...)
end

Base.iterate(F::WoodburyPDFactorization) = (F.L, Val(:R))
Base.iterate(F::WoodburyPDFactorization, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::WoodburyPDFactorization, ::Val{:done}) = nothing

Base.size(F::WoodburyPDFactorization, i::Int...) = size(F.U, i...)

function Base.Matrix{S}(F::WoodburyPDFactorization{T}) where {S,T}
    return convert(Matrix{S}, lmul!(F.L, Matrix{T}(F.R)))
end
Base.Matrix(F::WoodburyPDFactorization{T}) where {T} = Matrix{T}(F)

function Base.inv(F::WoodburyPDFactorization)
    return WoodburyPDFactorization(inv(F.U'), F.Q, inv(F.V'))
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::WoodburyPDFactorization)
    summary(io, F)
    println(io)
    println(io, "R factor:")
    return Base.show(io, mime, F.R)
end

LinearAlgebra.transpose(F::WoodburyPDFactorization) = F
LinearAlgebra.adjoint(F::WoodburyPDFactorization) = F

function LinearAlgebra.lmul!(F::WoodburyPDFactorization, x::StridedVecOrMat)
    lmul!(F.R, x)
    lmul!(F.L, x)
    return x
end

function LinearAlgebra.ldiv!(F::WoodburyPDFactorization, x::StridedVecOrMat)
    ldiv!(F.L, x)
    ldiv!(F.R, x)
    return x
end

LinearAlgebra.det(F::WoodburyPDFactorization) = (det(F.U) * det(F.V))^2
function LinearAlgebra.logabsdet(F::WoodburyPDFactorization)
    l = 2 * (logdet(F.U) + logdet(F.V))
    return (l, one(l))
end

"""
    WoodburyPDRightFactor{T,TA,Q,TC} <: AbstractMatrix{T}

The right factor ``R`` of a [`WoodburyPDFactorization`](@ref).
"""
struct WoodburyPDRightFactor{
    T<:Real,
    TU<:Union{Diagonal{T},LinearAlgebra.AbstractTriangular{T}},
    TQ, # wide type to support any Q
    TV<:Union{Diagonal{T},LinearAlgebra.AbstractTriangular{T}},
} <: AbstractMatrix{T}
    U::TU
    Q::TQ
    V::TV
end

const WoodburyPDLeftFactor{T,U,Q,V} = LinearAlgebra.AdjOrTrans{
    T,WoodburyPDRightFactor{T,U,Q,V}
}

Base.size(R::WoodburyPDRightFactor, dims::Int...) = size(R.U, dims...)

function Base.Matrix{S}(R::WoodburyPDRightFactor{T}) where {S,T}
    return convert(Matrix{S}, lmul!(R, Matrix{T}(I, size(R))))
end

Base.copy(R::WoodburyPDRightFactor) = Matrix(R)

Base.getindex(R::WoodburyPDRightFactor, i::Int, j::Int) = getindex(copy(R), i, j)

function Base.inv(R::WoodburyPDRightFactor)
    return transpose(WoodburyPDRightFactor(inv(R.U'), R.Q, inv(R.V')))
end

function LinearAlgebra.mul!(
    r::StridedVecOrMat{T}, R::WoodburyPDRightFactor{T}, x::StridedVecOrMat{T}
) where {T}
    copyto!(r, x)
    return lmul!(R, copyto!(r, x))
end

function Base.:*(R::WoodburyPDRightFactor, x::StridedVecOrMat)
    T = Base.promote_eltype(R, x)
    y = copyto!(similar(x, T), x)
    return lmul!(R, y)
end

function LinearAlgebra.lmul!(R::WoodburyPDRightFactor, x::StridedVecOrMat)
    k = min(size(R.U, 1), size(R.V, 1))
    lmul!(R.U, x)
    lmul!(R.Q', x)
    @views lmul!(R.V, x isa AbstractVector ? x[1:k] : x[1:k, :])
    return x
end
function LinearAlgebra.lmul!(L::WoodburyPDLeftFactor, x::StridedVecOrMat)
    R = parent(L)
    k = min(size(R.U, 1), size(R.V, 1))
    @views lmul!(R.V', x isa AbstractVector ? x[1:k] : x[1:k, :])
    lmul!(R.Q, x)
    lmul!(R.U', x)
    return x
end

function Base.:\(F::Union{WoodburyPDLeftFactor,WoodburyPDRightFactor}, x::StridedVecOrMat)
    T = Base.promote_eltype(F, x)
    y = copyto!(similar(x, T), x)
    return ldiv!(F, y)
end

function LinearAlgebra.ldiv!(R::WoodburyPDRightFactor, x::StridedVecOrMat)
    k = min(size(R.U, 1), size(R.V, 1))
    @views ldiv!(R.V, x isa AbstractVector ? x[1:k] : x[1:k, :])
    lmul!(R.Q, x)
    ldiv!(R.U, x)
    return x
end
function LinearAlgebra.ldiv!(L::WoodburyPDLeftFactor, x::StridedVecOrMat)
    R = parent(L)
    k = min(size(R.U, 1), size(R.V, 1))
    ldiv!(R.U', x)
    lmul!(R.Q', x)
    @views ldiv!(R.V', x isa AbstractVector ? x[1:k] : x[1:k, :])
    return x
end

LinearAlgebra.det(R::WoodburyPDRightFactor) = det(R.V) * det(R.Q) * det(R.U)
function LinearAlgebra.logabsdet(R::WoodburyPDRightFactor)
    lQ, s = logabsdet(R.Q)
    return (logdet(R.V) + lQ + logdet(R.U), s)
end

"""
    pdfactorize(A, B, D) -> WoodburyPDFactorization

Factorize the positive definite matrix ``W = A + B D B^\\mathrm{T}``.

The result is the "square root" factorization `(L, R)`, where ``W = L R`` and
``L = R^\\mathrm{T}``.

Let ``U^\\mathrm{T} U = A`` be the Cholesky decomposition of ``A``, and let
``Q X = U^{-\\mathrm{T}} B`` be a thin QR decomposition. Define ``C = I + XDX^\\mathrm{T}``,
with the Cholesky decomposition ``V^\\mathrm{T} V = C``. Then, ``W = R^\\mathrm{T} R``,
where
```math
R = \\begin{pmatrix} U & 0 \\\\ 0 & I \\end{pmatrix} Q^\\mathrm{T} V.
```

The positive definite requirement is equivalent to the requirement that both ``A`` and
``C`` are positive definite.

For a derivation of this decomposition for the special case of diagonal ``A``, see
appendix A of [^Zhang2021].

[^Zhang2021]: Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).
                Pathfinder: Parallel quasi-Newton variational inference.
                arXiv: [2108.03782](https://arxiv.org/abs/2108.03782) [stat.ML]

See [`pdunfactorize`](@ref), [`WoodburyPDFactorization`](@ref), [`WoodburyPDMat`](@ref)
"""
function pdfactorize(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix)
    cholA = cholesky(A isa Diagonal ? A : Symmetric(A))
    U = cholA.U
    Q, R = qr(U' \ B)
    V = cholesky(Symmetric(muladd(R, D * R', I))).U
    return WoodburyPDFactorization(U, Q, V)
end

"""
    pdunfactorize(F::WoodburyPDFactorization) -> (A, B, D)

Perform a reverse operation to [`pdfactorize`](@ref).

Note that this function does not compute the inverse of `pdfactorize`. It only computes
matrices that produce the same matrix ``W = A + B D B^\\mathrm{T}`` as for the inputs to
`pdfactorize`.
"""
function pdunfactorize(F::WoodburyPDFactorization)
    A = F.U' * F.U
    B = F.U' * Matrix(F.Q)
    D = muladd(F.V', F.V, -I)
    return A, B, D
end

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
and ``D`` must be chosen such that ``W`` is positive definite; otherwise an error will be
thrown during construction.

Upon construction, `WoodburyPDMat` calls [`pdfactorize`](@ref) to construct a
[`WoodburyPDFactorization`](@ref), which is used in its overloads.
z
See [`pdfactorize`](@ref), [`WoodburyPDFactorization`](@ref)
"""
struct WoodburyPDMat{
    T<:Real,
    TA<:AbstractMatrix{T},
    TB<:AbstractMatrix{T},
    TD<:AbstractMatrix{T},
    TF<:WoodburyPDFactorization{T},
} <: PDMats.AbstractPDMat{T}
    A::TA
    B::TB
    D::TD
    F::TF
end

function WoodburyPDMat(
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    return WoodburyPDMat(A, B, D, pdfactorize(A, B, D))
end
function WoodburyPDMat(A, B, D)
    T = Base.promote_eltype(A, B, D)
    return WoodburyPDMat(
        convert(AbstractMatrix{T}, A),
        convert(AbstractMatrix{T}, B),
        convert(AbstractMatrix{T}, D),
    )
end

pdfactorize(A::WoodburyPDMat) = A.F

LinearAlgebra.factorize(A::WoodburyPDMat) = pdfactorize(A)

Base.Matrix(W::WoodburyPDMat) = Matrix(Symmetric(muladd(W.B, W.D * W.B', W.A)))

function Base.AbstractMatrix{T}(W::WoodburyPDMat) where {T}
    F = pdfactorize(W)
    Fnew = WoodburyPDFactorization(
        convert(AbstractMatrix{T}, F.U),
        convert(AbstractMatrix{T}, F.Q),
        convert(AbstractMatrix{T}, F.V),
    )
    return WoodburyPDMat(
        convert(AbstractMatrix{T}, W.A),
        convert(AbstractMatrix{T}, W.B),
        convert(AbstractMatrix{T}, W.D),
        Fnew,
    )
end

function Base.getindex(W::WoodburyPDMat, i::Int, j::Int)
    B = W.B
    isempty(B) && return W.A[i, j]
    D = W.D isa Diagonal ? W.D : Symmetric(W.D)
    return @views W.A[i, j] + dot(B[i, :], D, B[j, :])
end

Base.adjoint(W::WoodburyPDMat) = W

Base.transpose(W::WoodburyPDMat) = W

function Base.inv(W::WoodburyPDMat)
    F = inv(W.F)
    A, B, D = pdunfactorize(F)
    return WoodburyPDMat(A, B, D, F)
end

LinearAlgebra.det(W::WoodburyPDMat) = det(factorize(W))
LinearAlgebra.logabsdet(W::WoodburyPDMat) = logabsdet(factorize(W))

function LinearAlgebra.diag(W::WoodburyPDMat)
    D = W.D isa Diagonal ? W.D : Symmetric(W.D)
    return diag(W.A) + map(b -> dot(b, D, b), eachrow(W.B))
end

# workaround for PDMats requiring `.dim` property only here
# see https://github.com/JuliaStats/PDMats.jl/pull/170
function Base.:+(a::WoodburyPDMat, b::LinearAlgebra.UniformScaling)
    return a + PDMats.ScalMat(size(a, 1), b.λ)
end
function Base.:+(a::LinearAlgebra.UniformScaling, b::WoodburyPDMat)
    return PDMats.ScalMat(size(b, 1), a.λ) + b
end

function LinearAlgebra.lmul!(W::WoodburyPDMat, x::AbstractVecOrMat)
    return lmul!(factorize(W), x)
end

function LinearAlgebra.mul!(y::AbstractVector, W::WoodburyPDMat, x::AbstractVecOrMat)
    return lmul!(W, copyto!(y, x))
end

function Base.:*(W::WoodburyPDMat, c::Real)
    c > 0 || return Matrix(W) * c
    return WoodburyPDMat(W.A * c, W.B * one(c), W.D * c)
end

function Base.size(W::WoodburyPDMat)
    n = size(W.A, 1)
    return (n, n)
end

PDMats.dim(W::WoodburyPDMat) = size(W.A, 1)

function PDMats.invquad(W::WoodburyPDMat, x::AbstractVector{T}) where {T}
    return sum(abs2, pdfactorize(W).L \ x)
end

function PDMats.invquad!(r::AbstractArray, W::WoodburyPDMat, x::AbstractMatrix{T}) where {T}
    v = pdfactorize(W).L \ x
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad!(r::AbstractArray, W::WoodburyPDMat, x::AbstractMatrix{T}) where {T}
    v = pdfactorize(W).R * x
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad(W::WoodburyPDMat, x::AbstractVector{T}) where {T}
    v = pdfactorize(W).R * x
    return sum(abs2, v)
end

function PDMats.unwhiten!(
    r::AbstractVecOrMat{T}, W::WoodburyPDMat, x::AbstractVecOrMat{T}
) where {T}
    copyto!(r, x)
    return lmul!(pdfactorize(W).L, r)
end

function invunwhiten!(
    r::AbstractVecOrMat{T}, W::WoodburyPDMat, x::AbstractVecOrMat{T}
) where {T}
    copyto!(r, x)
    return ldiv!(pdfactorize(W).R, r)
end

# adapted from https://github.com/JuliaStats/PDMats.jl/blob/master/src/utils.jl
function colwise_sumsq!(r::AbstractArray, a::AbstractMatrix)
    eachindex(r) == axes(a, 2) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    for j in axes(a, 2)
        v = zero(eltype(a))
        @simd for i in axes(a, 1)
            @inbounds v += abs2(a[i, j])
        end
        r[j] = v
    end
    return r
end
