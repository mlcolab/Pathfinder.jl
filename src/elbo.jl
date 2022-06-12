struct MaximumELBO{save_draws,R,E}
    rng::R
    ndraws::Int
    executor::E
end
function MaximumELBO(;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    ndraws::Int=DEFAULT_NDRAWS_ELBO,
    executor=Transducers.SequentialEx(),
    save_draws::Bool=false,
)
    return MaximumELBO{save_draws,typeof(rng),typeof(executor)}(rng, ndraws, executor)
end

function (optimizer::MaximumELBO{R,E,save_draws})(logp, _, _, dists) where {R,E,save_draws}
    (; rng, ndraws, executor) = optimizer
    EE = Core.Compiler.return_type(
        _compute_elbo, Tuple{typeof(rng),typeof(logp),eltype(dists),Int}
    )
    dists_not_init = @views dists[(begin + 1):end]
    estimates = similar(dists_not_init, EE)
    isempty(estimates) && return false, first(dists), 0, estimates
    Folds.map!(estimates, dists_not_init, executor) do dist
        return _compute_elbo(rng, logp, dist, ndraws)
    end
    _, iteration_opt = _findmax(estimates |> Transducers.Map(est -> est.value))
    elbo_opt = estimates[iteration_opt].value
    success = !isnan(elbo_opt) & (elbo_opt != -Inf)
    return success, dists[iteration_opt + 1], iteration_opt, estimates
end

function _compute_elbo(rng, logp, dist, ndraws)
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

function Base.show(io::IO, ::MIME"text/plain", elbo::ELBOEstimate)
    print(io, "ELBO estimate: ", _to_string(elbo))
    return nothing
end

function _to_string(est::ELBOEstimate; digits=2)
    return "$(round(est.value; digits)) ± $(round(est.std_err; digits))"
end
