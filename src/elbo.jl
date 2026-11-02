function maximize_elbo(rng, logp, dists, ndraws, ntasks::Int; save_samples::Bool=true)
    seeds = rand!(rng, similar(dists, UInt64))
    # each chunk gets its own RNG and draws buffer. `elbo_and_samples!` copies the draws if
    # they are to be saved, so the buffer can be reused.
    setup = () -> (copy(rng), _elbo_draws_buffer(dists, ndraws))
    estimates = _chunk_tmap(dists, seeds; ntasks, setup) do (chunk_rng, draws), dist, seed
        Random.seed!(chunk_rng, seed)
        return elbo_and_samples!(draws, chunk_rng, logp, dist; save_samples)
    end
    isempty(estimates) && return 0, estimates
    _, iteration_opt = _findmax_skipnan(est -> est.value, estimates)
    return iteration_opt, estimates
end

# not inlined into the `setup` closure of `maximize_elbo`, since a type captured by a closure
# is stored as a `DataType`, which breaks inference on older Julia versions
function _elbo_draws_buffer(dists, ndraws)
    dim = isempty(dists) ? 0 : length(first(dists))
    return Matrix{eltype(eltype(dists))}(undef, dim, ndraws)
end

function elbo_and_samples!(ϕ, rng, logp, dist; save_samples::Bool=true, ntasks::Int=1)
    ϕ, logqϕ = rand_and_logpdf!(rng, dist, ϕ)
    # draws are generated sequentially above, so only `logp` is evaluated in parallel
    logpϕ = _maybe_tmap!(logp, similar(logqϕ), eachcol(ϕ); ntasks)
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr; mean = elbo) / length(logr))
    ϕ_save = save_samples ? copyto!(similar(ϕ), ϕ) : similar(ϕ, map(zero, size(ϕ)))
    return ELBOEstimate(elbo, elbo_se, ϕ_save, logpϕ, logqϕ, logr)
end

struct ELBOEstimate{T,P<:AbstractMatrix{T},L<:Vector{T}}
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
