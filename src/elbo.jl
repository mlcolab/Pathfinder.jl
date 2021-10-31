function maximize_elbo(rng, logp, dists, ndraws)
    elbo_ϕ_logqϕ = map(dists) do dist
        elbo_and_samples(rng, logp, dist, ndraws)
    end
    lopt = argmax(first.(elbo_ϕ_logqϕ))
    return (lopt, elbo_ϕ_logqϕ[lopt]...)
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = logp.(eachcol(ϕ))
    elbo = elbo_from_logpdfs(logpϕ, logqϕ)
    return elbo, ϕ, logqϕ
end

elbo_from_logpdfs(logpϕ, logqϕ) = Statistics.mean(logpϕ) - Statistics.mean(logqϕ)
