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

    result = pathfinder(result; ndraws=1_000)
    @test result isa PathfinderResult
    @test result.input === model
    @test size(result.draws) == (5, 1_000)
    @test result.draws_transformed isa MCMCChains.Chains

    result = multipathfinder(result, 500; nruns=5)
    @test result isa MultiPathfinderResult
    @test result.input === model
    @test size(result.draws) == (5, 500)
    @test result.draws_transformed isa MCMCChains.Chains
end
