function maximize_elbo(rng, logp, dists, ndraws, executor)
    elbo_ϕ_logqϕ1 = elbo_and_samples(rng, logp, dists[begin], ndraws)
    elbo_ϕ_logqϕ = similar(dists, typeof(elbo_ϕ_logqϕ1))
    elbo_ϕ_logqϕ[begin] = elbo_ϕ_logqϕ1
    iter =
        eachindex(dists, elbo_ϕ_logqϕ)[(begin + 1):end] |> Transducers.Map() do i
            elbo_ϕ_logqϕ[i] = elbo_and_samples(rng, logp, dists[i], ndraws)
            return nothing
        end
    Folds.collect(iter, executor)
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
