using AdvancedHMC,
    ForwardDiff,
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

Random.seed!(2)

struct RegressionProblem{X,Y}
    x::X
    y::Y
end

function (prob::RegressionProblem)(θ)
    (; σ, α, β) = θ
    (; x, y) = prob
    lp = normlogpdf(σ) + logtwo
    lp += normlogpdf(α)
    lp += sum(normlogpdf, β)
    y_hat = muladd(x, β, α)
    lp += sum(eachindex(y_hat, y)) do i
        return normlogpdf(y_hat[i], σ, y[i])
    end
    return lp
end

function as_draw_array(θs)
    draw_param_mat = transpose(reduce(hcat, θs))
    return reshape(draw_param_mat, size(draw_param_mat, 1), 1, size(draw_param_mat, 2))
end

function mean_and_mcse(θs)
    ms = dropdims(mean(θs; dims=(1, 2)); dims=(1, 2))
    ses = mcse(θs)
    return ms, ses
end

function compare_estimates(xs1, xs2, α=0.01)
    nparams = length(first(xs1))
    α /= nparams  # bonferroni correction
    p = α / 2
    m1, s1 = mean_and_mcse(xs1)
    m2, s2 = mean_and_mcse(xs2)
    zs = @. (m1 - m2) / hypot(s1, s2)
    @test all(norminvcdf(p) .< zs .< norminvccdf(p))
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

        @testset "WoodburyPDMat constructor" begin
            m = AdvancedHMC.RankUpdateEuclideanMetric(M⁻¹)
            @test m isa AdvancedHMC.RankUpdateEuclideanMetric
            @test m.A == A
            @test m.B == B
            @test m.D == D
        end

        metric2 = Pathfinder.RankUpdateEuclideanMetric(3)
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (3,)
        @test size(metric2, 2) == 1
        @test eltype(metric2) === Float64
        metric2 = Pathfinder.RankUpdateEuclideanMetric((4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4,)
        @test size(metric2, 2) == 1
        metric2 = Pathfinder.RankUpdateEuclideanMetric(Float32, (4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4,)
        @test size(metric2, 2) == 1
        @test eltype(metric2.M⁻¹) === Float32

        @test size(metric) == (5,)
        @test size(metric, 2) == 1
        @test AdvancedHMC.renew(metric, M⁻¹2).M⁻¹ === M⁻¹2
        @test sprint(show, metric) == "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))"
        @test AdvancedHMC.neg_energy(h, r, θ) ≈ AdvancedHMC.neg_energy(h_dense, r, θ)
        @test AdvancedHMC.∂H∂r(h, r) ≈ AdvancedHMC.∂H∂r(h_dense, r)
        kinetic = AdvancedHMC.GaussianKinetic()
        rng = Random.default_rng()
        if isdefined(AdvancedHMC, :rand_momentum)  # AHMC ≥ v0.7.0
            compare_estimates(
                as_draw_array([
                    AdvancedHMC.rand_momentum(rng, metric, kinetic, θ) for _ in 1:10_000
                ]),
                as_draw_array([
                    AdvancedHMC.rand_momentum(rng, metric_dense, kinetic, θ) for
                    _ in 1:10_000
                ]),
            )
        else
            compare_estimates(
                as_draw_array([rand(rng, metric, kinetic) for _ in 1:10_000]),
                as_draw_array([rand(rng, metric_dense, kinetic) for _ in 1:10_000]),
            )
        end
    end

    @testset "sample" begin
        ndraws = 10_000
        nadapts = 500
        nparams = 5
        x = 0:0.01:1
        y = sin.(x) .+ randn.() .* 0.2 .+ x
        X = [x x .^ 2 x .^ 3]
        θ₀ = randn(nparams)
        prob = RegressionProblem(X, y)
        trans = as((σ=asℝ₊, α=asℝ, β=as(Array, size(X, 2))))
        P = TransformedLogDensity(trans, prob)
        ∇P = ADgradient(:ForwardDiff, P)

        metric = DiagEuclideanMetric(nparams)
        hamiltonian = Hamiltonian(metric, ∇P)
        ϵ = find_good_stepsize(hamiltonian, θ₀)
        integrator = Leapfrog(ϵ)
        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )
        samples1, stats1 = sample(
            hamiltonian,
            kernel,
            θ₀,
            ndraws + nadapts,
            adaptor,
            nadapts;
            drop_warmup=true,
            progress=false,
        )

        result_pf = pathfinder(∇P; dim=5, optimizer=Optim.LBFGS(; m=6))

        @testset "Initial point" begin
            metric = DiagEuclideanMetric(nparams)
            hamiltonian = Hamiltonian(metric, ∇P)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples2, stats2 = sample(
                hamiltonian,
                kernel,
                result_pf.draws[:, 1],
                ndraws + nadapts,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(as_draw_array(samples2), as_draw_array(samples1))
        end

        @testset "Initial point and metric" begin
            metric = DiagEuclideanMetric(diag(result_pf.fit_distribution.Σ))
            hamiltonian = Hamiltonian(metric, ∇P)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples3, stats3 = sample(
                hamiltonian,
                kernel,
                result_pf.draws[:, 1],
                ndraws + nadapts,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(as_draw_array(samples3), as_draw_array(samples1))
        end

        @testset "Initial point and final metric" begin
            metric = Pathfinder.RankUpdateEuclideanMetric(result_pf.fit_distribution.Σ)
            hamiltonian = Hamiltonian(metric, ∇P)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples4, stats4 = sample(
                hamiltonian,
                kernel,
                result_pf.draws[:, 1],
                ndraws + nadapts,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(as_draw_array(samples4), as_draw_array(samples1))
        end
    end
end
