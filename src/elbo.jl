function maximize_elbo(rng, logp, dists, ndraws, ntasks::Int)
    seeds = rand!(rng, similar(dists, UInt64))
    estimates = _chunk_tmap(dists, seeds; ntasks, setup=() -> copy(rng)) do chunk_rng, dist, seed
        Random.seed!(chunk_rng, seed)
        return elbo_and_samples(chunk_rng, logp, dist, ndraws)
    end
    isempty(estimates) && return 0, estimates
    _, iteration_opt = _findmax_skipnan(est -> est.value, estimates)
    return iteration_opt, estimates
end

function elbo_and_samples(rng, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr; mean = elbo) / length(logr))
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

function Base.show(io::IO, ::MIME"text/plain", elbo::ELBOEstimate)
    print(io, "ELBO estimate: ", _to_string(elbo))
    return nothing
end

function _to_string(est::ELBOEstimate; digits=2)
    return "$(round(est.value; digits)) ± $(round(est.std_err; digits))"
end
