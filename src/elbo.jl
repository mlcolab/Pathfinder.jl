function maximize_elbo(rng, logp, dists, ndraws, executor)
    estimate = elbo_and_samples(rng, logp, dists[begin], ndraws)
    estimates = similar(dists, typeof(estimate))
    estimates[begin] = estimate
    @views Folds.map!(estimates[(begin + 1):end], dists[(begin + 1):end], executor) do dist
        return elbo_and_samples(rng, logp, dist, ndraws)
    end
    _, lopt = _findmax(estimates |> Transducers.Map(est -> est.value))
    return lopt, estimates[lopt]
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr) / length(logr))
    return ELBOEstimate(elbo, elbo_se, ϕ, logpϕ, logqϕ, logr)
end

struct ELBOEstimate{T,P,L<:AbstractVector{T}}
    value::T
    std_err::T
    draws::P
    log_densities_target::L
    log_densities_fit::L
    log_density_ratios::L
end

function Base.show(io::IO, elbo::ELBOEstimate)
    print(
        io,
        "ELBO estimate ",
        round(elbo.value; digits=2),
        " ± ",
        round(elbo.std_err; digits=2),
    )
    return nothing
end
