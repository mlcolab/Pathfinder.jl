using Distributions
using LinearAlgebra
using Pathfinder
using PSIS
using Random
using Test

@testset "_resample" begin
    dim = 3
    ndraws_per_component = 10
    ncomponents = 4
    ndraws = 20
    rng = Xoshiro(42)
    draws_per_component = randn(rng, dim, ndraws_per_component, ncomponents)
    draws_all = reshape(draws_per_component, dim, :)

    @testset "uniform sampling (nothing)" begin
        draws, component_ids = Pathfinder._resample(rng, draws_per_component, nothing, ndraws)
        @test draws isa AbstractMatrix
        @test size(draws) == (dim, ndraws)
        @test component_ids isa AbstractVector{Int}
        @test length(component_ids) == ndraws
        @test all(1 .≤ component_ids .≤ ncomponents)
        for col in eachcol(draws)
            @test any(==(col), eachcol(draws_all))
        end
    end

    @testset "without replacement" begin
        draws, _ = Pathfinder._resample(rng, draws_per_component, nothing, 5; replace=false)
        @test length(unique(eachcol(draws))) == 5
    end

    @testset "weighted sampling (PSISResult)" begin
        # weight only the first component
        log_weights = fill(-1_000.0, ndraws_per_component, ncomponents)
        log_weights[:, 1] .= 0.0
        psis_result = PSIS.psis(vec(log_weights))
        draws, component_ids = Pathfinder._resample(rng, draws_per_component, psis_result, ndraws)
        @test size(draws) == (dim, ndraws)
        @test all(==(1), component_ids)
        for col in eachcol(draws)
            @test any(==(col), eachcol(draws_per_component[:, :, 1]))
        end
    end

    @testset "component_ids consistent with draws" begin
        draws, component_ids = Pathfinder._resample(rng, draws_per_component, nothing, ndraws)
        for (draw, cid) in zip(eachcol(draws), component_ids)
            @test 1 ≤ cid ≤ ncomponents
            @test any(==(draw), eachcol(draws_per_component[:, :, cid]))
        end
    end
end

@testset "_compute_log_importance_ratios" begin
    dim = 2
    ndraws_per_component = 5
    ncomponents = 3
    rng = Xoshiro(7)

    μs = [fill(float(k), dim) for k in 1:ncomponents]
    components = [MvNormal(μ, I(dim)) for μ in μs]
    draws_per_component = stack(map(c -> rand(rng, c, ndraws_per_component), components))

    target = MvNormal(zeros(dim), I(dim))
    logp(x) = logpdf(target, x)

    log_ratios = Pathfinder._compute_log_importance_ratios(logp, components, draws_per_component, 1)
    @test log_ratios isa AbstractVector{<:Real}
    @test length(log_ratios) == ndraws_per_component * ncomponents

    # Each draw x at position (j, k) should satisfy: ratio = logp(x) - logpdf(components[k], x).
    # vec iterates column-major: (j=1,k=1), …, (j=N,k=1), (j=1,k=2), …
    expected = vec([
        logp(draws_per_component[:, j, k]) - logpdf(components[k], draws_per_component[:, j, k])
        for j in 1:ndraws_per_component, k in 1:ncomponents
    ])
    @test log_ratios ≈ expected
end

@testset "_compute_psis_result" begin
    dim = 2
    ndraws_per_component = 20
    ncomponents = 3
    rng = Xoshiro(11)

    components = [MvNormal(fill(float(k), dim), I(dim)) for k in 1:ncomponents]
    draws_per_component = stack(map(c -> rand(rng, c, ndraws_per_component), components))

    target = MvNormal(zeros(dim), I(dim))
    logp(x) = logpdf(target, x)

    psis_result = Pathfinder._compute_psis_result(logp, components, draws_per_component; ntasks=1)
    @test psis_result isa PSIS.PSISResult
    @test length(psis_result.weights) == ndraws_per_component * ncomponents
    @test sum(psis_result.weights) ≈ 1
end

@testset "_candidate_draws_and_psis_result" begin
    dim = 3
    ndraws_per_component = 8
    nruns = 3
    logp(x) = -sum(abs2, x) / 2
    ℓ = build_logdensityproblem(logp, dim, 2)
    rng = Xoshiro(99)

    result_with_psis = multipathfinder(ℓ, ndraws_per_component; nruns, ndraws_per_run=ndraws_per_component, rng)
    result_no_psis = multipathfinder(ℓ, ndraws_per_component; nruns, ndraws_per_run=ndraws_per_component, rng, importance=false)
    @test result_with_psis.psis_result isa PSIS.PSISResult
    @test result_no_psis.psis_result === nothing

    @testset "::Nothing reuses stored draws" begin
        draws, psis = Pathfinder._candidate_draws_and_psis_result(rng, result_with_psis, nothing)
        @test size(draws) == (dim, ndraws_per_component, nruns)
        @test psis === result_with_psis.psis_result
        expected = stack(map(r -> r.draws, result_with_psis.pathfinder_results))
        @test draws == expected
    end

    @testset "::Nothing with no stored PSIS returns nothing" begin
        draws, psis = Pathfinder._candidate_draws_and_psis_result(rng, result_no_psis, nothing)
        @test size(draws) == (dim, ndraws_per_component, nruns)
        @test psis === nothing
    end

    @testset "::Int generates fresh draws" begin
        ndraws_new = 15
        draws, psis = Pathfinder._candidate_draws_and_psis_result(rng, result_with_psis, ndraws_new)
        @test size(draws) == (dim, ndraws_new, nruns)
        @test psis === nothing
    end
end
