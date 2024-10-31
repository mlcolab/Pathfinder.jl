function maximize_elbo(rng, logp, dists, ndraws, executor; save_samples::Bool=true)
    dim = isempty(dists) ? 0 : length(first(dists))
    draws = Matrix{eltype(eltype(dists))}(undef, (dim, ndraws))
    EE = Core.Compiler.return_type(
        elbo_and_samples!, Tuple{typeof(draws),typeof(rng),typeof(logp),eltype(dists)}
    )
    estimates = similar(dists, EE)
    isempty(estimates) && return 0, estimates

    Folds.map!(estimates, dists, executor) do dist
        return elbo_and_samples!(draws, rng, logp, dist; save_samples)
    end
    _, iteration_opt = _findmax(estimates |> Transducers.Map(est -> est.value))
    return iteration_opt, estimates
end

function elbo_and_samples!(ϕ, rng, logp, dist; save_samples::Bool=true)
    ϕ, logqϕ = rand_and_logpdf!(rng, dist, ϕ)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr) / length(logr))
    ϕ_save = save_samples ? copyto!(similar(ϕ), ϕ) : similar(ϕ, map(zero, size(ϕ)))
    return ELBOEstimate(elbo, elbo_se, ϕ_save, logpϕ, logqϕ, logr)
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
