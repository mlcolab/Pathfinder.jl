module Pathfinder

using Random, LinearAlgebra, Statistics

using Optim: Optim, LineSearches
using PSIS
using StatsBase
using StatsFuns
using WoodburyMatrices: SymWoodbury

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate Hessian positive-definite
const DEFAULT_OPTIMIZER = Optim.LBFGS(; m=5, linesearch=LineSearches.MoreThuente())

function pathfinder(
    logp,
    ∇logp,
    θ₀,
    ndraws;
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer::Optim.AbstractOptimizer=DEFAULT_OPTIMIZER,
    history_length::Int=optimizer isa Optim.LBFGS ? optimizer.m : 5,
    ndraws_elbo::Int=5,
    kwargs...,
)
    θs, logpθs, ∇logpθs = optimize(logp, ∇logp, θ₀, optimizer; kwargs...)
    L = length(θs) - 1
    @assert length(logpθs) == length(∇logpθs) == L + 1
    μs, Σs = mvnormal_estimate(θs, ∇logpθs; history_length=history_length)
    ϕ_logqϕ_λ = map(μs, Σs) do μ, Σ
        ϕ, logqϕ = mvnormal_sample_logpdf(rng, μ, Σ, ndraws_elbo)
        λ = elbo(logp.(ϕ), logqϕ)
        return ϕ, logqϕ, λ
    end
    ϕ, logqϕ, λ = ntuple(i -> getindex.(ϕ_logqϕ_λ, i), Val(3))
    lopt = argmax(λ[2:end]) + 1
    @info "Optimized for $L iterations. Maximum ELBO of $(round(λ[lopt]; digits=2)) reached at iteration $(lopt - 1)."

    # get parameters of ELBO-maximizing distribution
    μopt = μs[lopt]
    Σopt = Σs[lopt]

    # reuse existing draws; draw additional ones if necessary
    ϕdraws = ϕ[lopt]
    logqϕdraws = logqϕ[lopt]
    if ndraws_elbo < ndraws
        append!.(
            (ϕdraws, logqϕdraws),
            mvnormal_sample_logpdf(rng, μopt, Σopt, ndraws - ndraws_elbo),
        )
    else
        ϕdraws = ϕdraws[1:ndraws]
        logqϕdraws = logqϕdraws[1:ndraws]
    end

    return μopt, Σopt, ϕdraws, logqϕdraws
end

# multipath-pathfinder
function multipathfinder(
    logp,
    ∇logp,
    θ₀s,
    ndraws;
    ndraws_per_run::Int=5,
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    if ndraws > ndraws_per_run * length(θ₀s)
        @warn "More draws requested than total number of draws across replicas. Draws will not be unique."
    end
    # TODO: allow to be parallelized
    res = map(θ₀s) do θ₀
        μ, Σ, ϕ, logqϕ = pathfinder(logp, ∇logp, θ₀, ndraws_per_run; rng=rng, kwargs...)
        logpϕ = logp.(ϕ)
        return μ, Σ, ϕ, logpϕ - logqϕ
    end
    μs, Σs, ϕs, logqϕs = ntuple(i -> getindex.(res, i), Val(4))
    ϕsvec = reduce(vcat, ϕs)
    logqϕsvec = reduce(vcat, logqϕs)
    ϕsample = psir(rng, ϕsvec, logqϕsvec, ndraws)
    return ϕsample
end

function optimize(logp, ∇logp, θ₀, optimizer; kwargs...)
    f(x) = -logp(x)
    g!(y, x) = (y .= .-∇logp(x))

    options = Optim.Options(; store_trace=true, extended_trace=true, kwargs...)
    res = Optim.optimize(f, g!, θ₀, optimizer, options)

    θ = Optim.minimizer(res)
    θs = Optim.x_trace(res)::Vector{typeof(θ)}
    logpθs = -Optim.f_trace(res)
    ∇logpθs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))::typeof(θs)
    return θs, logpθs, ∇logpθs
end

elbo(logpϕ, logqϕ) = mean(logpϕ) - mean(logqϕ)

function psir(rng, ϕ, log_ratios, ndraws)
    logw, _ = PSIS.psis(log_ratios; normalize=true)
    w = StatsBase.pweights(exp.(logw))
    return StatsBase.sample(rng, ϕ, w, ndraws; replace=true)
end

# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function mvnormal_estimate(θs, ∇logpθs; history_length=5, ϵ=1e-12)
    L = length(θs) - 1
    θ = θs[1]
    N = length(θ)
    s = similar(θ)
    # S = similar(θ, N, history_length)
    S = Vector{typeof(s)}(undef, 0)

    ∇logpθ = ∇logpθs[1]
    y = similar(∇logpθ)
    # Y = similar(∇logpθ, N, history_length)
    Y = Vector{typeof(y)}(undef, 0)

    α, β, γ = fill!(similar(θ), true), similar(θ, N, 0), similar(θ, 0, 0)
    Σ = SymWoodbury(Diagonal(α), β, γ)
    μs = [muladd(Σ, ∇logpθ, θ)]
    Σs = [Σ]

    m = 0
    for l in 1:L
        s .= θs[l + 1] .- θs[l]
        y .= ∇logpθs[l] .- ∇logpθs[l + 1]
        α′ = copy(α)
        b = dot(y, s)
        if b > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            # replace oldest stored s and y with new ones
            push!(S, copy(s))
            push!(Y, copy(y))
            m += 1

            if length(S) > history_length
                popfirst!(S)
                popfirst!(Y)
            end

            # Gilbert et al, eq 4.9
            a = dot(y, Diagonal(α), y)
            c = dot(s, inv(Diagonal(α)), s)
            @. α′ = b / (a / α + y^2 - (a / c) * (s / α)^2)
            α = α′
        else
            @warn "Skipping inverse Hessian update to avoid negative curvature."
        end

        J′ = length(S) # min(m, history_length)
        β = similar(θ, N, 2J′)
        γ = fill!(similar(θ, 2J′, 2J′), false)
        for j in 1:J′
            yⱼ = Y[j]
            sⱼ = S[j]
            β[1:N, j] .= α .* yⱼ
            β[1:N, J′ + j] .= sⱼ
            for i in 1:(j - 1)
                γ[J′ + i, J′ + j] = dot(S[i], yⱼ)
            end
            γ[J′ + j, J′ + j] = dot(sⱼ, yⱼ)
        end
        R = @views UpperTriangular(γ[(J′ + 1):(2J′), (J′ + 1):(2J′)])
        nRinv = @views UpperTriangular(γ[1:J′, (J′ + 1):(2J′)])
        copyto!(nRinv, -I)
        ldiv!(R, nRinv)
        nRinv′ = @views LowerTriangular(copyto!(γ[(J′ + 1):(2J′), 1:J′], nRinv'))
        for j in 1:J′
            αyⱼ = β[1:N, j]
            for i in 1:(j - 1)
                γ[J′ + i, J′ + j] = dot(Y[i], αyⱼ)
            end
            γ[J′ + j, J′ + j] += dot(Y[j], αyⱼ)
        end
        γ22 = @view γ[(J′ + 1):(2J′), (J′ + 1):(2J′)]
        LinearAlgebra.copytri!(γ22, 'U', false, false)
        rmul!(γ22, nRinv)
        lmul!(nRinv′, γ22)

        Σ = SymWoodbury(Diagonal(α), β, γ)
        push!(μs, muladd(Σ, ∇logpθ, θ))
        push!(Σs, Σ)
    end
    return μs, Σs
end

function mvnormal_sample_logpdf(rng, μ, Σ::SymWoodbury, ndraws)
    N = length(μ)

    # draw points
    u = Random.randn!(rng, similar(μ, N, ndraws))
    unormsq = map(x -> sum(abs2, x), eachcol(u))
    x = unwhiten!(Σ, u)
    x .+= μ

    # compute log density at each point
    logpx = ((logabsdet(Σ)[1] + N * log2π) .+ unormsq) ./ -2

    return collect(eachcol(x)), logpx
end

# given x drawn from an IID standard normal, in-place map x to one drawn from a
# zero-centered normal with covariance Σ
function unwhiten!(Σ::SymWoodbury, x)
    α = Σ.A.diag
    β = Σ.B
    γ = Σ.D

    # compute components for T, where Σ=TTᵀ, i.e. T = √α Q [L 0; 0 I]
    F = qr(β ./ sqrt.(α))
    R = UpperTriangular(F.R)
    Z = rmul!(R * γ, R')
    Z[diagind(Z)] .+= true
    L = cholesky(Symmetric(Z)).L

    # apply x ↦ Tx
    J = size(β, 2)
    @views lmul!(L, x[1:J])
    lmul!(F.Q, x)
    x .*= sqrt.(α)

    return x
end

end
