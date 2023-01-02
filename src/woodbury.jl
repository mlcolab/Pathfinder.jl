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
    TQ, # wide type to support any Q
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

function Base.getindex(W::WoodburyPDMat, i::Int, j::Int)
    B = W.B
    isempty(B) && return W.A[i, j]
    return @views W.A[i, j] + dot(B[i, :], W.D, B[j, :])
end

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

# workaround for PDMats requiring `.dim` property only here
# see https://github.com/JuliaStats/PDMats.jl/pull/170
function Base.:+(a::WoodburyPDMat, b::LinearAlgebra.UniformScaling)
    return a + PDMats.ScalMat(size(a, 1), b.λ)
end
function Base.:+(a::LinearAlgebra.UniformScaling, b::WoodburyPDMat)
    return PDMats.ScalMat(size(b, 1), a.λ) + b
end

function LinearAlgebra.lmul!(W::WoodburyPDMat, x::AbstractVecOrMat)
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
    v = W.Q' * (W.UA' \ x)
    n, m = size(W.B)
    k = min(m, n)
    return @views sum(abs2, W.UC' \ v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.invquad!(r::AbstractArray, W::WoodburyPDMat, x::AbstractMatrix{T}) where {T}
    v = lmul!(W.Q', W.UA' \ x)
    k = minimum(size(W.B))
    @views ldiv!(W.UC', v[1:k, :])
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad!(r::AbstractArray, W::WoodburyPDMat, x::AbstractMatrix{T}) where {T}
    v = lmul!(W.Q', W.UA * x)
    k = minimum(size(W.B))
    @views lmul!(W.UC, v[1:k, :])
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad(W::WoodburyPDMat, x::AbstractVector{T}) where {T}
    v = W.Q' * (W.UA * x)
    n, m = size(W.B)
    k = min(m, n)
    return @views sum(abs2, W.UC * v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.unwhiten!(
    r::AbstractVecOrMat{T}, W::WoodburyPDMat, x::AbstractVecOrMat{T}
) where {T}
    k = minimum(size(W.B))
    copyto!(r, x)
    @views lmul!(W.UC', x isa AbstractVector ? r[1:k] : r[1:k, :])
    lmul!(W.Q, r)
    lmul!(W.UA', r)
    return r
end

function invunwhiten!(
    r::AbstractVecOrMat{T}, W::WoodburyPDMat, x::AbstractVecOrMat{T}
) where {T}
    k = minimum(size(W.B))
    copyto!(r, x)
    @views ldiv!(W.UC, x isa AbstractVector ? r[1:k] : r[1:k, :])
    lmul!(W.Q, r)
    ldiv!(W.UA, r)
    return r
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
