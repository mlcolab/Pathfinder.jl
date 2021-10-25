using LinearAlgebra

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
