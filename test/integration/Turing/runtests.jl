using LogDensityProblems, Pathfinder, Random, Test, Turing
using Turing.Bijectors

if isdefined(Base, :get_extension)
    PathfinderTuringExt = Base.get_extension(Pathfinder, :PathfinderTuringExt)
else
    PathfinderTuringExt = Pathfinder.PathfinderTuringExt
end

Random.seed!(0)

#! format: off
@model function regression_model(x, y)
    σ ~ truncated(Normal(); lower=0)
    α ~ Normal()
    β ~ filldist(Normal(), size(x, 2))
    y_hat = muladd(x, β, α)
    y .~ Normal.(y_hat, σ)
    return (; y)
end

# adapted from https://github.com/TuringLang/Turing.jl/issues/2195
@model function dynamic_const_model()
    lb ~ Uniform(0, 0.1)
    ub ~ Uniform(0.11, 0.2)
    x ~ Bijectors.transformed(
        Normal(0, 1), Bijectors.inverse(Bijectors.Logit(lb, ub))
    )
end

@model function transformed_model(dist, bijector)
    y ~ Bijectors.transformed(dist, bijector)
end
#! format: on

@testset "Turing integration" begin
    @testset "create_log_density_problem" begin
        @testset for bijector in [elementwise(log), Bijectors.SimplexBijector()],
            udist in [Normal(1, 2), Normal(3, 4)],
            n in [1, 5]

            binv = Bijectors.inverse(bijector)
            dist = filldist(udist, n)
            dist_trans = Bijectors.transformed(dist, binv)
            model = transformed_model(dist, binv)
            prob = PathfinderTuringExt.create_log_density_problem(model)
            @test LogDensityProblems.capabilities(prob) isa
                LogDensityProblems.LogDensityOrder{0}
            x = rand(n, 10)
            # after applying the Jacobian correction, the log-density of the model should
            # be the same as the log-density of the distribution in unconstrained space
            @test LogDensityProblems.logdensity.(Ref(prob), eachcol(x)) ≈ logpdf(dist, x)
        end
    end

    @testset "draws_to_chains" begin
        draws = randn(3, 100)
        model = dynamic_const_model()
        chns = PathfinderTuringExt.draws_to_chains(model, draws)
        @test chns isa MCMCChains.Chains
        @test size(chns) == (100, 3, 1)
        @test names(chns) == [:lb, :ub, :x]
        @test all(0 .< chns[:, :lb, 1] .< 0.1)
        @test all(0.11 .< chns[:, :ub, 1] .< 0.2)
        @test all(chns[:, :lb, 1] .< chns[:, :x, 1] .< chns[:, :ub, 1])
    end

    @testset "integration tests" begin
        @testset "regression model" begin
            x = 0:0.01:1
            y = sin.(x) .+ randn.() .* 0.2 .+ x
            X = [x x .^ 2 x .^ 3]
            model = regression_model(X, y)
            expected_param_names = Symbol.(["α", "β[1]", "β[2]", "β[3]", "σ"])

            result = pathfinder(model; ndraws=10_000)
            @test result isa PathfinderResult
            @test result.input === model
            @test size(result.draws) == (5, 10_000)
            @test result.draws_transformed isa MCMCChains.Chains
            @test result.draws_transformed.info.pathfinder_result isa PathfinderResult
            @test sort(names(result.draws_transformed)) == expected_param_names
            @test all(>(0), result.draws_transformed[:σ])
            init_params = Vector(result.draws_transformed.value[1, :, 1])
            chns = sample(model, NUTS(), 10_000; init_params, progress=false)
            @test mean(chns).nt.mean ≈ mean(result.draws_transformed).nt.mean rtol = 0.1

            result = multipathfinder(model, 10_000; nruns=4)
            @test result isa MultiPathfinderResult
            @test result.input === model
            @test size(result.draws) == (5, 10_000)
            @test length(result.pathfinder_results) == 4
            @test result.draws_transformed isa MCMCChains.Chains
            @test result.draws_transformed.info.pathfinder_result isa MultiPathfinderResult
            @test sort(names(result.draws_transformed)) == expected_param_names
            @test all(>(0), result.draws_transformed[:σ])
            init_params = Vector(result.draws_transformed.value[1, :, 1])
            chns = sample(model, NUTS(), 10_000; init_params, progress=false)
            @test mean(chns).nt.mean ≈ mean(result.draws_transformed).nt.mean rtol = 0.1
        end
    end
end
