function maximize_elbo(rng, logp, dists, ndraws)
    elbo_ϕ_logqϕ = map(dists) do dist
        elbo_and_samples(rng, logp, dist, ndraws)
    end
    lopt = _argmax_ignore_nan(first.(elbo_ϕ_logqϕ))
    return (lopt, elbo_ϕ_logqϕ[lopt]...)
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = logp.(eachcol(ϕ))
    elbo = elbo_from_logpdfs(logpϕ, logqϕ)
    return elbo, ϕ, logqϕ
end

elbo_from_logpdfs(logpϕ, logqϕ) = Statistics.mean(logpϕ) - Statistics.mean(logqϕ)

function _argmax_ignore_nan(x)
    imax, _ = foldl(pairs(x)) do (i1, x1), (i2, x2)
        return (isnan(x2) || !Base.isless(x1, x2)) ? (i1, x1) : (i2, x2)
    end
    return imax
end
