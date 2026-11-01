"""
    fit_mvnormals(θs, ∇logpθs; history_length=5) -> (dists, num_bfgs_updates_rejected)

Fit a multivariate-normal distribution to each point on the trajectory `θs`.

Given `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian approximations with
the provided `history_length`. The inverse Hessians approximate a covariance. The
covariances and corresponding means that define multivariate normal approximations per
point are returned.

The 2nd returned value is the number of BFGS updates to the inverse Hessian matrices that
were rejected due to keeping the inverse Hessian positive definite.
"""
function fit_mvnormals(θs, ∇logpθs; kwargs...)
    Σs, num_bfgs_updates_rejected = lbfgs_inverse_hessians(θs, ∇logpθs; kwargs...)
    dists = map(fit_mvnormal, Σs, θs, ∇logpθs)
    return dists, num_bfgs_updates_rejected
end

"""
    fit_mvnormal(Σ, θ, ∇logpθ) -> Distributions.MvNormal
    fit_mvnormal(state::LBFGSInverseHessianState) -> Distributions.MvNormal

Fit a multivariate normal distribution at the point `θ` with gradient `∇logpθ`, taking the
inverse Hessian approximation `Σ` as its covariance; or at the current point of `state`.
"""
function fit_mvnormal(Σ, θ, ∇logpθ)
    μ = muladd(Σ, ∇logpθ, θ)
    return Distributions.MvNormal(μ, Σ)
end
function fit_mvnormal(state::LBFGSInverseHessianState)
    return fit_mvnormal(lbfgs_inverse_hessian(state), state.θ, state.∇logpθ)
end

# faster than computing `logpdf` and `rand` independently
function rand_and_logpdf(rng, dist::Distributions.MvNormal, ndraws)
    μ = dist.μ
    Σ = dist.Σ
    N = length(μ)

    # draw points
    u = Random.randn!(rng, similar(μ, N, ndraws))
    unormsq = vec(sum(abs2, u; dims=1))
    x = PDMats.unwhiten!(u, Σ, u)
    x .+= μ

    # compute log density at each point
    logpx = (muladd(N, log2π, logdet(Σ)) .+ unormsq) ./ -2

    return x, logpx
end
