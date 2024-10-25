using DynamicHMC,
    LinearAlgebra,
    LogDensityProblems,
    LogDensityProblemsAD,
    MCMCDiagnosticTools,
    Optim,
    Pathfinder,
    Random,
    Statistics,
    StatsFuns,
    Test,
    TransformVariables
using TransformedLogDensities: TransformedLogDensity

Random.seed!(1)

struct RegressionProblem{X,Y}
    x::X
    y::Y
end

function (prob::RegressionProblem)(θ)
    σ = θ.σ
    α = θ.α
    β = θ.β
    x = prob.x
    y = prob.y
    lp = normlogpdf(σ) + logtwo
    lp += normlogpdf(α)
    lp += sum(normlogpdf, β)
    y_hat = muladd(x, β, α)
    lp += sum(eachindex(y_hat, y)) do i
        return normlogpdf(y_hat[i], σ, y[i])
    end
    return lp
end

function mean_and_mcse(θs)
    ms = dropdims(mean(θs; dims=(1, 2)); dims=(1, 2))
    ses = mcse(θs)
    return ms, ses
end

function compare_estimates(xs1, xs2, α=0.01)
    nparams = first(size(xs1))
    α /= nparams  # bonferroni correction
    p = α / 2
    m1, s1 = mean_and_mcse(xs1)
    m2, s2 = mean_and_mcse(xs2)
    zs = @. (m1 - m2) / hypot(s1, s2)
    @test all(norminvcdf(p) .< zs .< norminvccdf(p))
end

@testset "DynamicHMC integration" begin
    A = Diagonal(rand(5))
    B = randn(5, 2)
    D = exp(Symmetric(randn(2, 2)))
    M⁻¹ = Pathfinder.WoodburyPDMat(A, B, D)

    @testset "DynamicHMC.GaussianKineticEnergy" begin
        κ = DynamicHMC.GaussianKineticEnergy(M⁻¹)
        @test κ.M⁻¹ === M⁻¹
        @test κ.W isa Transpose{T,<:Pathfinder.WoodburyPDRightFactor{T}} where {T}
        @test κ.W * κ.W' ≈ inv(M⁻¹)
        κdense = DynamicHMC.GaussianKineticEnergy(Symmetric(Matrix(M⁻¹)))
        κdense2 = DynamicHMC.GaussianKineticEnergy(Symmetric(Matrix(M⁻¹)), Matrix(κ.W))
        p = randn(5)
        q = randn(5)

        @test size(κ) == size(κdense)
        @test DynamicHMC.kinetic_energy(κ, p, q) ≈ DynamicHMC.kinetic_energy(κdense, p, q)
        @test DynamicHMC.calculate_p♯(κ, p, q) ≈ DynamicHMC.calculate_p♯(κdense, p, q)
        @test DynamicHMC.∇kinetic_energy(κ, p, q) ≈ DynamicHMC.∇kinetic_energy(κdense, p, q)
        @test DynamicHMC.rand_p(MersenneTwister(42), κ, q) ≈
            DynamicHMC.rand_p(MersenneTwister(42), κdense2, q)
    end

    @testset "DynamicHMC.mcmc_with_warmup" begin
        ndraws = 10_000
        x = 0:0.01:1
        y = sin.(x) .+ randn.() .* 0.2 .+ x
        X = [x x .^ 2 x .^ 3]
        prob = RegressionProblem(X, y)
        trans = as((σ=asℝ₊, α=asℝ, β=as(Array, size(X, 2))))
        P = TransformedLogDensity(trans, prob)
        ∇P = ADgradient(:ForwardDiff, P)
        rng = Random.GLOBAL_RNG

        result_hmc1 = mcmc_with_warmup(rng, ∇P, ndraws; reporter=NoProgressReport())

        result_pf = pathfinder(∇P)

        @testset "Initial point" begin
            result_hmc2 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(; q=result_pf.draws[:, 1]),
                reporter=NoProgressReport(),
            )
            @test result_hmc2.κ.M⁻¹ isa Diagonal
            compare_estimates(
                DynamicHMC.stack_posterior_matrices([result_hmc2]),
                DynamicHMC.stack_posterior_matrices([result_hmc1]),
            )
        end

        @testset "Initial point and metric" begin
            result_hmc3 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(;
                    q=result_pf.draws[:, 1],
                    κ=GaussianKineticEnergy(result_pf.fit_distribution.Σ),
                ),
                warmup_stages=default_warmup_stages(; M=Symmetric),
                reporter=NoProgressReport(),
            )
            @test result_hmc3.κ.M⁻¹ isa Symmetric
            compare_estimates(
                DynamicHMC.stack_posterior_matrices([result_hmc3]),
                DynamicHMC.stack_posterior_matrices([result_hmc1]),
            )
        end

        @testset "Initial point and final metric" begin
            result_hmc4 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(;
                    q=result_pf.draws[:, 1],
                    κ=GaussianKineticEnergy(result_pf.fit_distribution.Σ),
                ),
                warmup_stages=default_warmup_stages(; middle_steps=0, doubling_stages=0),
                reporter=NoProgressReport(),
            )
            @test result_hmc4.κ.M⁻¹ === result_pf.fit_distribution.Σ
            compare_estimates(
                DynamicHMC.stack_posterior_matrices([result_hmc4]),
                DynamicHMC.stack_posterior_matrices([result_hmc1]),
            )
        end
    end
end
