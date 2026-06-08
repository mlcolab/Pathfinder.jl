function maximize_elbo(rng, logp, dists, ndraws, ntasks::Int)
    seeds = rand!(rng, similar(dists, UInt64))
    copy_rng = copy(rng)
    nchunks = min(length(dists), ntasks)
    estimates = if nchunks > 1 && Threads.nthreads() > 1
        nrngs = min(nchunks, Threads.nthreads())
        rng_pool = Channel{typeof(copy_rng)}(nrngs)
        put!(rng_pool, copy_rng)
        for _ in 2:nrngs
            put!(rng_pool, copy(rng))
        end
        OhMyThreads.tmapreduce(
            vcat,
            OhMyThreads.chunks(eachindex(dists, seeds); n=nchunks);
            chunking=false,
        ) do chunk
            chunk_rng = take!(rng_pool)
            try
                return map(chunk) do i
                    Random.seed!(chunk_rng, seeds[i])
                    return elbo_and_samples(chunk_rng, logp, dists[i], ndraws)
                end
            finally
                put!(rng_pool, chunk_rng)
            end
        end
    else
        map(dists, seeds) do dist, seed
            Random.seed!(copy_rng, seed)
            return elbo_and_samples(copy_rng, logp, dist, ndraws)
        end
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
