using LinearAlgebra
using LogDensityProblems
using Optim
using OptimizationNLopt
using Pathfinder
using ProgressLogging
using SciMLBase
using Test

include("test_utils.jl")

@testset "build_optim_function" begin
    n = 20
    ℓ = build_logdensityproblem(logp_banana, n)
    x = randn(n)

    @testset for optimizer in (Optim.LBFGS(), Optim.Newton())
        fun = @inferred Pathfinder.build_optim_function(ℓ, optimizer)
        @test fun isa Pathfinder.OptimJLFunction
        @test fun.prob === ℓ
    end

    @testset for optimizer in (NLopt.Opt(:LD_LBFGS, n),)
        fun = @inferred Pathfinder.build_optim_function(ℓ, optimizer)
        @test fun isa SciMLBase.OptimizationFunction
        @test SciMLBase.isinplace(fun)
        @test fun.f(x, nothing) ≈ -ℓ.logp(x)
        ∇fx = similar(x)
        fun.grad(∇fx, x, nothing)
        @test ∇fx ≈ -ℓ.∇logp(x)
        H = similar(x, n, n)
        fun.hess(H, x, nothing)
        @test H ≈ -ℓ.∇²logp(x)
    end
end

@testset "build_optim_problem" begin
    n = 20
    ℓ = build_logdensityproblem(logp_banana, n)
    x0 = randn(n)

    @testset "OptimJLFunction" begin
        fun = Pathfinder.OptimJLFunction(ℓ)
        prob = @inferred Pathfinder.build_optim_problem(fun, x0)
        @test prob isa Pathfinder.OptimJLProblem
        @test prob.fun === fun
        @test prob.u0 === x0
    end

    @testset "SciML.OptimizationFunction" begin
        fun = Pathfinder._build_sciml_optim_function(ℓ)
        prob = Pathfinder.build_optim_problem(fun, x0)
        @test SciMLBase.isinplace(prob)
        @test prob isa SciMLBase.OptimizationProblem
        @test prob.f === fun
        @test prob.u0 == x0
        @test prob.p === nothing
    end
end

@testset "optimize_with_trace" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2
    ℓ = build_logdensityproblem(f, n)

    x0 = randn(n)

    optimizers = [
        "Optim.BFGS" => Optim.BFGS(),
        "Optim.LBFGS" => Optim.LBFGS(),
        "Optim.ConjugateGradient" => Optim.ConjugateGradient(),
        "NLopt.Opt" => NLopt.Opt(:LD_LBFGS, n),
    ]
    @testset "$name" for (name, optimizer) in optimizers
        fun = Pathfinder.build_optim_function(ℓ, optimizer)
        prob = Pathfinder.build_optim_problem(fun, x0)
        optim_sol, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer)
        @test optim_trace isa Pathfinder.OptimizationTrace
        @test optim_trace.points[1] ≈ x0
        @test optim_trace.points[end] ≈ μ
        @test optim_trace.log_densities ≈ f.(optim_trace.points)
        @test optim_trace.gradients ≈ ℓ.∇logp.(optim_trace.points) atol = 1e-4

        if optimizer isa Union{Optim.FirstOrderOptimizer,Optim.SecondOrderOptimizer}
            @test optim_sol isa Optim.MultivariateOptimizationResults

            options = Optim.Options(; store_trace=true, extended_trace=true)
            res = Optim.optimize(
                x -> -f(x), (y, x) -> copyto!(y, -ℓ.∇logp(x)), x0, optimizer, options
            )
            @test Optim.iterations(res) == Optim.iterations(optim_sol)
            @test Optim.x_trace(res) ≈ Optim.x_trace(optim_sol)
            @test Optim.minimizer(res) ≈ Optim.minimizer(optim_sol)
        else
            @test optim_sol isa SciMLBase.OptimizationSolution
            @test optim_sol.u ≈ μ
        end

        @testset "progress logging" begin
            @testset for try_id in 1:2
                logs, = Test.collect_test_logs(; min_level=ProgressLogging.ProgressLevel) do
                    ProgressLogging.progress(; name="Optimizing") do progress_id
                        Pathfinder.optimize_with_trace(prob, Optim.LBFGS(); progress_id, try_id)
                    end
                end
                @test logs[1].kwargs[:progress] === nothing
                @test logs[1].message == "Optimizing"
                @test logs[2].kwargs[:progress] == 0.0
                @test logs[2].message == "Optimizing (try $try_id)"
                @test logs[3].kwargs[:progress] == 0.001
                @test logs[end].kwargs[:progress] == "done"
            end
        end
    end
end
