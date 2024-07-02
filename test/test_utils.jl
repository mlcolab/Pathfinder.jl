using ForwardDiff
using LinearAlgebra
using LogDensityProblems
using Pathfinder
using Random

function rand_pd_mat(rng, T, n)
    U = qr(randn(rng, T, n, n)).Q
    return Matrix(Symmetric(U * rand_pd_diag_mat(rng, T, n) * U'))
end
rand_pd_mat(T, n) = rand_pd_mat(Random.GLOBAL_RNG, T, n)

rand_pd_diag_mat(rng, T, n) = Diagonal(rand(rng, T, n))
rand_pd_diag_mat(T, n) = rand_pd_diag_mat(Random.GLOBAL_RNG, T, n)

# defined for testing purposes
function Pathfinder.rand_and_logpdf(rng, dist, ndraws)
    x = rand(rng, dist, ndraws)
    if x isa AbstractVector
        xmat = permutedims(x)
        logpx = Distributions.logpdf.(dist, x)
    else
        xmat = x
        logpx = Distributions.logpdf(dist, x)
    end
    return xmat, logpx
end

# banana distribution
_ϕb(b, x) = [x[1]; x[2] + b * (x[1]^2 - 100); x[3:end]]
function _fb(b, x)
    y = _ϕb(b, x)
    Σ = Diagonal([100; ones(length(y) - 1)])
    return -dot(y, inv(Σ), y) / 2
end
logp_banana(x) = _fb(0.03, x)

struct LogDensityFunction{F,G,H}
    logp::F
    dim::Int
    ∇logp::G
    ∇²logp::H
end
LogDensityFunction(logp, dim) = LogDensityFunction(logp, dim, nothing, nothing)
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{<:Any,G,H}}
) where {G,H}
    order = !(G <: Nothing) + !(H <: Nothing)
    return LogDensityProblems.LogDensityOrder{order}()
end
LogDensityProblems.dimension(ℓ::LogDensityFunction) = ℓ.dim
LogDensityProblems.logdensity(ℓ::LogDensityFunction, x) = ℓ.logp(x)
function LogDensityProblems.logdensity_and_gradient(ℓ::LogDensityFunction, x)
    return ℓ.logp(x), ℓ.∇logp(x)
end
function LogDensityProblems.logdensity_gradient_and_hessian(ℓ::LogDensityFunction, x)
    return ℓ.logp(x), ℓ.∇logp(x), ℓ.∇²logp(x)
end

function build_logdensityproblem(f, n, max_order)
    ∇f = max_order > 0 ? Base.Fix1(ForwardDiff.gradient, f) : nothing
    Hf = max_order > 1 ? Base.Fix1(ForwardDiff.hessian, f) : nothing
    ℓ = LogDensityFunction(f, n, ∇f, Hf)
    return ℓ
end
