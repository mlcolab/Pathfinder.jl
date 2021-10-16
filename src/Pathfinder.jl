module Pathfinder

using Random, LinearAlgebra, Statistics

using Optim: Optim, LineSearches
using PSIS
using StatsBase
using StatsFuns

export pathfinder, multipathfinder

function pathfinder(
    logp,
    ∇logp,
    θ₀;
    rng = Random.default_rng(),
    J = 5,
    L = 1_000,
    K = 5,
    M = 5,
    kwargs...,
)
    θs, logpθs, ∇logpθs = lbfgs(logp, ∇logp, θ₀; J = J, L = L, kwargs...)
    L = length(θs) - 1
    @assert length(logpθs) == length(∇logpθs) == L + 1
    αs, βs, γs = cov_estimate(θs, ∇logpθs; J = J)
    ϕ_logqϕ_λ = map(θs, ∇logpθs, αs, βs, γs) do θ, ∇logpθ, α, β, γ
        ϕ, logqϕ = bfgs_sample(rng, θ, ∇logpθ, α, β, γ, K)
        λ = elbo(logp.(ϕ), logqϕ)
        return ϕ, logqϕ, λ
    end
    ϕ, logqϕ, λ = ntuple(i -> getindex.(ϕ_logqϕ_λ, i), Val(3))
    lopt = argmax(λ[2:end]) + 1
    @info "Optimized for $L iterations. Maximum ELBO of $(round(λ[lopt]; digits=2)) reached at iteration $(lopt - 1)."

    μopt = θs[lopt] .+ αs[lopt] .* ∇logpθs[lopt]
    Σopt = Diagonal(αs[lopt]) + βs[lopt] * γs[lopt] * βs[lopt]'
    return μopt, Σopt, ϕ[lopt], logqϕ[lopt]
end

# multipath-pathfinder
function multipathfinder(
    logp,
    ∇logp,
    θ₀s;
    R = length(θ₀s),
    rng = Random.default_rng(),
    kwargs...,
)
    # TODO: allow to be parallelized
    res = map(θ₀s) do θ₀
        μ, Σ, ϕ, logqϕ = pathfinder(logp, ∇logp, θ₀; rng = rng, kwargs...)
        logpϕ = logp.(ϕ)
        return μ, Σ, ϕ, logpϕ - logqϕ
    end
    μs, Σs, ϕs, logws = ntuple(i -> getindex.(res, i), Val(4))
    ϕsvec = reduce(vcat, ϕs)
    logwsvec = reduce(vcat, logws)
    ϕsample = psir(rng, ϕsvec, logwsvec, R)
    return ϕsample
end

function lbfgs(logp, ∇logp, θ₀; J = 5, L = 1_000, ϵ = 2.2e-16, kwargs...)
    f(x) = -logp(x)
    g!(y, x) = (y .= .-∇logp(x))

    options = Optim.Options(;
        store_trace = true,
        extended_trace = true,
        iterations = L,
        kwargs...,
    )
    optimizer = Optim.LBFGS(; m = J, linesearch = LineSearches.MoreThuente())
    res = Optim.optimize(f, g!, θ₀, optimizer, options)

    θ = Optim.minimizer(res)
    θs = Optim.x_trace(res)::Vector{typeof(θ)}
    logpθs = -Optim.f_trace(res)
    ∇logpθs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))::typeof(θs)
    return θs, logpθs, ∇logpθs
end

elbo(logpϕ, logqϕ) = mean(logpϕ) - mean(logqϕ)

function psir(rng, ϕ, log_ratios, R)
    logw, _ = PSIS.psis(log_ratios; normalize = true)
    w = StatsBase.pweights(exp.(logw))
    return StatsBase.sample(rng, ϕ, w, R; replace = true)
end

# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function cov_estimate(θs, ∇logpθs; J = 5, ϵ = 1e-12)
    L = length(θs) - 1
    θ = θs[1]
    N = length(θ)
    s = similar(θ)
    # S = similar(θ, N, J)
    S = Vector{typeof(s)}(undef, 0)

    ∇logpθ = ∇logpθs[1]
    y = similar(∇logpθ)
    # Y = similar(∇logpθ, N, J)
    Y = Vector{typeof(y)}(undef, 0)

    α, β, γ = fill!(similar(θ), true), similar(θ, N, 0), similar(θ, 0, 0)
    αs = [α]
    βs = [β]
    γs = [γ]

    m = 0
    for l = 1:L
        s .= θs[l+1] .- θs[l]
        y .= ∇logpθs[l] .- ∇logpθs[l+1]
        α′ = copy(α)
        b = dot(y, s)
        if b > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            # replace oldest stored s and y with new ones
            push!(S, copy(s))
            push!(Y, copy(y))
            m += 1

            if length(S) > J
                popfirst!(S)
                popfirst!(Y)
            end

            # Gilbert et al, eq 4.9
            a = dot(y, Diagonal(α), y)
            c = dot(s, Diagonal(inv.(α)), s)
            @. α′ = b / (a / α + y^2 - (a / c) * (s / α)^2)
            α = α′
        else
            @warn "Skipping inverse Hessian update to avoid negative curvature."
        end
        push!(αs, α)

        J′ = length(S) # min(m, J)
        β = similar(θ, N, 2J′)
        γ = fill!(similar(θ, 2J′, 2J′), false)
        for j = 1:J′
            yⱼ = Y[j]
            sⱼ = S[j]
            β[1:N, j] .= α .* yⱼ
            β[1:N, J′+j] .= sⱼ
            for i = 1:(j-1)
                γ[J′+i, J′+j] = dot(S[i], yⱼ)
            end
            γ[J′+j, J′+j] = dot(sⱼ, yⱼ)
        end
        R = @views UpperTriangular(γ[J′+1:2J′, J′+1:2J′])
        nRinv = @views UpperTriangular(γ[1:J′, J′+1:2J′])
        copyto!(nRinv, -I)
        ldiv!(R, nRinv)
        nRinv′ = @views LowerTriangular(copyto!(γ[J′+1:2J′, 1:J′], nRinv'))
        for j = 1:J′
            αyⱼ = β[1:N, j]
            for i = 1:(j-1)
                γ[J′+i, J′+j] = dot(Y[i], αyⱼ)
            end
            γ[J′+j, J′+j] += dot(Y[j], αyⱼ)
        end
        γ22 = @view γ[J′+1:2J′, J′+1:2J′]
        LinearAlgebra.copytri!(γ22, 'U', false, false)
        rmul!(γ22, nRinv)
        lmul!(nRinv′, γ22)

        push!(βs, β)
        push!(γs, γ)
    end
    return αs, βs, γs
end

function bfgs_sample(rng, θ, ∇logpθ, α, β, γ, M)
    N = length(θ)
    F = qr(β ./ sqrt.(α))
    Q = Matrix(F.Q)
    R = F.R
    L = cholesky(Symmetric(I + R * Symmetric(γ) * R')).L
    logdetΣ = sum(log, α) + 2logdet(L)
    μ = β * (γ * (β' * ∇logpθ))
    μ .+= θ .+ α .* ∇logpθ
    u = randn(rng, N, M)
    ϕ = μ .+ sqrt.(α) .* (Q * ((L - I) * (Q' * u)) .+ u)
    logqϕ = ((logdetΣ + N * log2π) .+ sum.(abs2, eachcol(u))) ./ -2
    return map(collect, eachcol(ϕ)), logqϕ
end

end
