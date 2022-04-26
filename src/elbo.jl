function maximize_elbo(rng, logp, dists, ndraws, executor)
    elbo_ϕ_logqϕ1 = elbo_and_samples(rng, logp, dists[begin], ndraws)
    elbo_ϕ_logqϕ = similar(dists, typeof(elbo_ϕ_logqϕ1))
    elbo_ϕ_logqϕ[begin] = elbo_ϕ_logqϕ1
    @views Folds.map!(
        elbo_ϕ_logqϕ[(begin + 1):end], dists[(begin + 1):end], executor
    ) do dist
        return elbo_and_samples(rng, logp, dist, ndraws)
    end
    _, lopt = _findmax(elbo_ϕ_logqϕ |> Transducers.Map(first))
    return (lopt, elbo_ϕ_logqϕ[lopt]...)
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    elbo = elbo_from_logpdfs(logpϕ, logqϕ)
    return elbo, ϕ, logqϕ
end

elbo_from_logpdfs(logpϕ, logqϕ) = Statistics.mean(logpϕ) - Statistics.mean(logqϕ)
