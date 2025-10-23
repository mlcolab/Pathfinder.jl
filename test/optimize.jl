using ADTypes
using LinearAlgebra
using Optim
using Optimization
using OptimizationNLopt
using Pathfinder
using ProgressLogging
using ReverseDiff
using SciMLBase
using Test
if isdefined(Optimization, :OptimizationState)
    using Optimization: OptimizationState
else
    using OptimizationBase: OptimizationState
end

@testset "build_optim_function" begin
    n = 20
    x = randn(n)

    @testset "callable" begin
        @testset for adtype in [SciMLBase.NoAD(), ADTypes.AutoForwardDiff()]
            fun = @inferred Pathfinder.build_optim_function(logp_banana, adtype)
            @test fun isa SciMLBase.OptimizationFunction
            @test fun.adtype === adtype
            @test fun.f(x) ≈ -logp_banana(x)
        end
    end

    @testset "log density problem" begin
        @testset for max_order in 0:2,
            adtype in [ADTypes.AutoForwardDiff(), ADTypes.AutoReverseDiff()]

            ℓ = build_logdensityproblem(logp_banana, n, max_order)
            ℓ_reference = build_logdensityproblem(logp_banana, n, 2)
            capabilities = LogDensityProblems.capabilities(ℓ)
            fun = Pathfinder.build_optim_function(ℓ, adtype, capabilities)
            @test fun isa SciMLBase.OptimizationFunction
            @test SciMLBase.isinplace(fun)
            @test fun.adtype === adtype
            @test fun.f(x, nothing) ≈ -ℓ.logp(x)
            ∇fx = similar(x)
            if max_order > 0
                fun.grad(∇fx, x, nothing)
                @test ∇fx ≈ -ℓ_reference.∇logp(x)
            end
            if max_order > 1
                H = similar(x, n, n)
                fun.hess(H, x, nothing)
                @test H ≈ -ℓ_reference.∇²logp(x)
            end
        end
    end
end

@testset "OptimizationCallback" begin
    @testset "callback return value" begin
        progress_name = "Optimizing"
        progress_id = nothing
        maxiters = 1_000
        x = randn(5)
        check_vals = [0.0, NaN, -Inf, Inf]
        @testset for fail_on_nonfinite in [true, false],
            fval in check_vals,
            gval in check_vals,
            cbfail in [true, false]

            xs = Vector{Float64}[]
            fxs = Float64[]
            ∇fxs = Union{Nothing,Vector{Float64}}[]
            ∇f = function (x)
                g = -x
                g[end] = gval
                return g
            end
            should_fail =
                cbfail ||
                (fail_on_nonfinite && (isnan(fval) || fval == Inf || !isfinite(gval)))

            callback = (state, args...) -> cbfail
            state = OptimizationState(; iter=0, u=x, objective=(-fval), grad=(-∇f(x)))
            cb_args = (state, -fval)

            cb = Pathfinder.OptimizationCallback(
                xs,
                fxs,
                ∇fxs,
                progress_name,
                progress_id,
                maxiters,
                callback,
                fail_on_nonfinite,
            )
            @test cb isa Pathfinder.OptimizationCallback
            @test cb(cb_args...) == should_fail
        end
    end
end

@testset "optimize_with_trace" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2

    x0 = randn(n)

    optimizers = [
        "Optim.BFGS" => Optim.BFGS(),
        "Optim.LBFGS" => Optim.LBFGS(),
        "Optim.ConjugateGradient" => Optim.ConjugateGradient(),
        "NLopt.Opt" => NLopt.Opt(:LD_LBFGS, n),
    ]
    @testset "$name" for (name, optimizer) in optimizers,
        (adtype, max_order) in [(SciMLBase.NoAD(), 1), (ADTypes.AutoForwardDiff(), 0)]

        ℓ = build_logdensityproblem(f, n, max_order)
        fun = Pathfinder.build_optim_function(ℓ, adtype, LogDensityProblems.capabilities(ℓ))
        prob = SciMLBase.OptimizationProblem(fun, x0)

        optim_sol, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer)
        ∇logp(x) = -optim_sol.cache.f.grad(similar(x), x)
        @test optim_sol isa SciMLBase.OptimizationSolution
        @test optim_trace isa Pathfinder.OptimizationTrace
        @test optim_trace.points[1] ≈ x0
        @test optim_trace.points[end] ≈ μ
        @test optim_sol.u ≈ μ
        @test optim_trace.log_densities ≈ f.(optim_trace.points)
        @test optim_trace.gradients ≈ ∇logp.(optim_trace.points) atol = 1e-4

        if !(optimizer isa NLopt.Opt)
            options = Optim.Options(; store_trace=true, extended_trace=true)
            res = Optim.optimize(
                x -> -f(x), (y, x) -> copyto!(y, -∇logp(x)), x0, optimizer, options
            )
            @test Optim.iterations(res) == length(optim_trace.points) - 1
            @test Optim.x_trace(res) ≈ optim_trace.points
            @test Optim.minimizer(res) ≈ optim_trace.points[end]
        end
    end

    @testset "progress logging" begin
        ℓ = build_logdensityproblem(f, n, 2)
        fun = Pathfinder.build_optim_function(
            ℓ, SciMLBase.NoAD(), LogDensityProblems.capabilities(ℓ)
        )
        prob = SciMLBase.OptimizationProblem(fun, x0)

        logs, = Test.collect_test_logs(; min_level=ProgressLogging.ProgressLevel) do
            ProgressLogging.progress(; name="Optimizing") do progress_id
                Pathfinder.optimize_with_trace(prob, Optim.LBFGS(); progress_id)
            end
        end
        @test logs[1].kwargs[:progress] === nothing
        @test logs[1].message == "Optimizing"
        @test logs[2].kwargs[:progress] == 0.0
        @test logs[2].message == "Optimizing"
        @test logs[3].kwargs[:progress] == 0.001
        @test logs[end].kwargs[:progress] == "done"
    end
end
