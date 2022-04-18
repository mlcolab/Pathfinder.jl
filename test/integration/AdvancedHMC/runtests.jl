using AdvancedHMC,
    Distributions, ForwardDiff, LinearAlgebra, Optim, Pathfinder, Random, StatsBase, Test

struct RegressionModel{X,Y}
    x::X
    y::Y
end

function (m::RegressionModel)(θ)
    logσ = θ[1]
    σ = exp(logσ)
    α = θ[2]
    β = θ[3:end]
    x = m.x
    y = m.y
    lp = logpdf(truncated(Normal(); lower=0), σ) + logσ
    lp += logpdf(Normal(), α)
    lp += sum(b -> logpdf(Normal(), b), β)
    y_hat = muladd(x, β, α)
    lp += sum(eachindex(y_hat, y)) do i
        return loglikelihood(Normal(y_hat[i], σ), y[i])
    end
    return lp
end

@testset "AdvancedHMC integration" begin
    A = Diagonal(rand(5))
    B = randn(5, 2)
    D = exp(Symmetric(randn(2, 2)))
    M⁻¹ = Pathfinder.WoodburyPDMat(A, B, D)
    A2 = Diagonal(rand(5))
    B2 = randn(5, 2)
    D2 = exp(Symmetric(randn(2, 2)))
    M⁻¹2 = Pathfinder.WoodburyPDMat(A2, B2, D2)
    ℓπ(x) = -sum(abs2, x) / 2
    ∂ℓπ∂θ(x) = -x

    @testset "RankUpdateEuclideanMetric" begin
        metric = Pathfinder.RankUpdateEuclideanMetric(M⁻¹)
        @test metric.M⁻¹ === M⁻¹
        metric_dense = AdvancedHMC.DenseEuclideanMetric(Symmetric(Matrix(M⁻¹)))
        h = AdvancedHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        h_dense = AdvancedHMC.Hamiltonian(metric_dense, ℓπ, ∂ℓπ∂θ)
        r = randn(5)
        θ = randn(5)

        metric2 = Pathfinder.RankUpdateEuclideanMetric(3)
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (3, 3)
        metric2 = Pathfinder.RankUpdateEuclideanMetric((4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4, 4)
        metric2 = Pathfinder.RankUpdateEuclideanMetric(Float32, (4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4, 4)
        @test eltype(metric2.M⁻¹) === Float32

        @test size(metric) == (5, 5)
        @test AdvancedHMC.renew(metric, M⁻¹2).M⁻¹ === M⁻¹2
        @test sprint(show, metric) == "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))"
        @test AdvancedHMC.neg_energy(h, r, θ) ≈ AdvancedHMC.neg_energy(h_dense, r, θ)
        @test AdvancedHMC.∂H∂r(h, r) ≈ AdvancedHMC.∂H∂r(h_dense, r)
        m, v = mean_and_var([rand(metric) for _ in 1:10_000])
        m_dense, v_dense = mean_and_var([rand(metric_dense) for _ in 1:10_000])
        # NOTE: a more strict test would compute MCSE here
        s = sqrt.(v .+ v_dense)
        bounds = quantile.(Normal.(0, s), 0.05)
        @test all(bounds .< m - m_dense .< -bounds)
    end

    @testset "sample" begin
        ndraws = 1_000
        nadapts = 500
        nparams = 5
        rng = MersenneTwister(42)
        x = 0:0.01:1
        y = sin.(x) .+ randn.(rng) .* 0.2 .+ x
        X = [x x .^ 2 x .^ 3]
        θ₀ = randn(rng, nparams)
        ℓπ = RegressionModel(X, y)

        metric = DiagEuclideanMetric(nparams)
        hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
        ϵ = find_good_stepsize(hamiltonian, θ₀)
        integrator = Leapfrog(ϵ)
        proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )
        samples1, stats1 = sample(
            rng,
            hamiltonian,
            proposal,
            θ₀,
            ndraws,
            adaptor,
            nadapts;
            drop_warmup=true,
            progress=false,
        )

        θ₀ = rand(rng, Uniform(-2, 2), nparams)
        result_pf = pathfinder(ℓπ, θ₀, 1; rng, optimizer=Optim.LBFGS(; m=6))
        θ₀ = result_pf[2][:, 1]
        metric = Pathfinder.RankUpdateEuclideanMetric(result_pf[1].Σ)
        hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
        ϵ = find_good_stepsize(hamiltonian, θ₀)
        integrator = Leapfrog(ϵ)
        proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        adaptor = StepSizeAdaptor(0.8, integrator)
        samples2, stats2 = sample(
            rng,
            hamiltonian,
            proposal,
            θ₀,
            ndraws,
            adaptor,
            nadapts;
            drop_warmup=true,
            progress=false,
        )

        # check that the posterior means are approximately equal
        m1, v1 = mean_and_var(samples1)
        m2, v2 = mean_and_var(samples2)
        # NOTE: a more strict test would compute MCSE here
        s = sqrt.(v1 .+ v2)
        m = m1 - m2
        bounds = quantile.(Normal.(0, s), 0.05)
        @test all(bounds .< m .< -bounds)
    end
end
