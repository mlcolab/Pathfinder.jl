function maximize_elbo(rng, logp, dists, ndraws, executor)
    EE = Core.Compiler.return_type(elbo_and_samples,Tuple{typeof(rng),typeof(logp),eltype(dists),Int})
    estimates = similar(dists, EE)
    isempty(estimates) && return 0, estimates
    Folds.map!(estimates, dists, executor) do dist
        return elbo_and_samples(rng, logp, dist, ndraws)
    end
    _, iteration_opt = _findmax(estimates |> Transducers.Map(est -> est.value))
    return iteration_opt, estimates
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
