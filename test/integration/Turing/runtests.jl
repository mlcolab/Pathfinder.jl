using Pathfinder, Random, Test, Turing

Random.seed!(0)

@model function regression_model(x, y)
    σ ~ truncated(Normal(); lower=0)
    α ~ Normal()
    β ~ filldist(Normal(), size(x, 2))
    y_hat = muladd(x, β, α)
    y .~ Normal.(y_hat, σ)
    return (; y)
end

@testset "Turing integration" begin
    x = 0:0.01:1
    y = sin.(x) .+ randn.() .* 0.2 .+ x
    X = [x x .^ 2 x .^ 3]
    model = regression_model(X, y)
    expected_param_names = Symbol.(["α", "β[1]", "β[2]", "β[3]", "σ"])

    result = pathfinder(model; ndraws=1_000)
    @test result isa PathfinderResult
    @test result.input === model
    @test size(result.draws) == (5, 1_000)
    @test result.draws_transformed isa MCMCChains.Chains
    @test result.draws_transformed.info.pathfinder_result isa PathfinderResult
    @test sort(names(result.draws_transformed)) == expected_param_names
    @test all(>(0), result.draws_transformed[:σ])

    result = multipathfinder(model, 500; nruns=5)
    @test result isa MultiPathfinderResult
    @test result.input === model
    @test size(result.draws) == (5, 500)
    @test result.draws_transformed isa MCMCChains.Chains
    @test result.draws_transformed.info.pathfinder_result isa MultiPathfinderResult
    @test sort(names(result.draws_transformed)) == expected_param_names
    @test all(>(0), result.draws_transformed[:σ])
end
