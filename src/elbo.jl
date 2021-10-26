function maximize_elbo(rng, logp, dists, ndraws)
    ϕ_logqϕ_λ = map(dists) do dist
        ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
        λ = elbo(logp.(ϕ), logqϕ)
        return ϕ, logqϕ, λ
    end
    ϕ, logqϕ, λ = ntuple(i -> getindex.(ϕ_logqϕ_λ, i), Val(3))
    lopt = argmax(λ)
    return lopt, ϕ, logqϕ, λ
end

elbo(logpϕ, logqϕ) = Statistics.mean(logpϕ) - Statistics.mean(logqϕ)
