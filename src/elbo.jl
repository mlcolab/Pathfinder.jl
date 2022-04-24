function maximize_elbo(rng, logp, dists, ndraws, executor)
    iter = dists |> Transducers.Map() do dist
        elbo_and_samples(rng, logp, dist, ndraws)
    end
    elbo_ϕ_logqϕ = Folds.collect(iter, executor)
    _, lopt = _findmax(elbo_ϕ_logqϕ |> Transducers.Map(first))
    return (lopt, elbo_ϕ_logqϕ[lopt]...)
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = eachcol(ϕ) |> Transducers.Map(logp) |> Transducers.tcollect
    elbo = elbo_from_logpdfs(logpϕ, logqϕ)
    return elbo, ϕ, logqϕ
end

elbo_from_logpdfs(logpϕ, logqϕ) = Statistics.mean(logpϕ) - Statistics.mean(logqϕ)
