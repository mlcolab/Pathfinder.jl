"""
    fit_mvnormals(θs, ∇logpθs; history_length=5)

Fit a multivariate-normal distribution to each point on the trajectory `θs`.

Given `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian approximations with
the provided `history_length`. The inverse Hessians approximate a covariance. The
covariances and corresponding means that define multivariate normal approximations per
point are returned.
"""
function fit_mvnormals(θs, ∇logpθs; kwargs...)
    Σs = lbfgs_inverse_hessians(θs, ∇logpθs; kwargs...)
    μs = muladd.(Σs, ∇logpθs, θs)
    return Distributions.MvNormal.(μs, Σs)
end

# faster than computing `logpdf` and `rand` independently
function rand_and_logpdf(rng, dist::Distributions.MvNormal, ndraws)
    μ = dist.μ
    Σ = dist.Σ
    N = length(μ)

    # draw points
    u = Random.randn!(rng, similar(μ, N, ndraws))
    unormsq = map(x -> sum(abs2, x), eachcol(u))
    x = PDMats.unwhiten!(u, Σ, u)
    x .+= μ

    # compute log density at each point
    logpx = (muladd(N, log2π, logdet(Σ)) .+ unormsq) ./ -2

    return collect(eachcol(x)), logpx
end
