using LinearAlgebra
using Pathfinder

function rand_pd_mat(T, n)
    U = qr(randn(T, n, n)).Q
    return Matrix(Symmetric(U * rand_pd_diag_mat(T, n) * U'))
end

rand_pd_diag_mat(T, n) = Diagonal(rand(T, n))

# defined for testing purposes
function Pathfinder.rand_and_logpdf(rng, dist, ndraws)
    x = rand(rng, dist, ndraws)
    if x isa AbstractVector
        logpx = Distributions.logpdf.(dist, x)
        xvec = x
    else
        logpx = Distributions.logpdf(dist, x)
        xvec = collect(eachcol(x))
    end
    return xvec, logpx
end

# banana distribution
_ϕb(b, x) = [x[1]; x[2] + b * (x[1]^2 - 100); x[3:end]]
function _fb(b, x)
    y = _ϕb(b, x)
    Σ = Diagonal([100; ones(length(y) - 1)])
    return -dot(y, inv(Σ), y) / 2
end
logp_banana(x) = _fb(0.03, x)
