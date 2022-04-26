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

function _argmax_ignore_nan(x)
    imax, _ = foldl(pairs(x)) do i1_x1, i2_x2
        x1 = last(i1_x1)
        x2 = last(i2_x2)
        return (isnan(x2) || !Base.isless(x1, x2)) ? i1_x1 : i2_x2
    end
    return imax
end
