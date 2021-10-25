function maximize_elbo(rng, logp, dists, ndraws)
    ϕ_logqϕ_λ = map(dists) do dist
        ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
        λ = elbo(logp.(ϕ), logqϕ)
        return ϕ, logqϕ, λ
    end
    ϕ, logqϕ, λ = ntuple(i -> getindex.(ϕ_logqϕ_λ, i), Val(3))
    lopt = argmax(λ[2:end]) + 1

    return lopt, ϕ[lopt], logqϕ[lopt], λ[lopt]
end

elbo(logpϕ, logqϕ) = mean(logpϕ) - mean(logqϕ)
