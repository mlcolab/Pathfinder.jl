using ADTypes
using LinearAlgebra
using Optim
using OptimizationNLopt
using Pathfinder
using ProgressLogging
using ReverseDiff
using SciMLBase
using Test

using OptimizationBase: OptimizationState

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
                nothing,
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

    @testset "online fit" begin
        ℓ = build_logdensityproblem(f, n, 1)
        fun = Pathfinder.build_optim_function(
            ℓ, SciMLBase.NoAD(), LogDensityProblems.capabilities(ℓ)
        )
        prob = SciMLBase.OptimizationProblem(fun, x0)
        logp(x) = -fun.f(x, nothing)
        history_length = 4
        ndraws_elbo = 3

        _, optim_trace = Pathfinder.optimize_with_trace(prob, Optim.LBFGS())
        dists, num_rejected = Pathfinder.fit_mvnormals(
            optim_trace.points, optim_trace.gradients; history_length
        )

        rng = MersenneTwister(11)
        online_fit = Pathfinder.OnlineFit(prob, logp; rng, ndraws_elbo, history_length)
        optim_sol, optim_trace_online = Pathfinder.optimize_with_trace(
            prob, Optim.LBFGS(); online_fit
        )
        @test optim_sol isa SciMLBase.OptimizationSolution
        # the trajectory is recorded as log-densities only
        @test optim_trace_online.log_densities == optim_trace.log_densities
        @test isempty(optim_trace_online.points)
        @test isempty(optim_trace_online.gradients)
        @test length(optim_trace_online) == length(optim_trace)
        # the L-BFGS state followed the same trajectory
        @test online_fit.lbfgs_state.θ ≈ optim_trace.points[end]
        @test online_fit.lbfgs_state.num_bfgs_updates_rejected == num_rejected
        @test Pathfinder.fit_mvnormal(online_fit.lbfgs_state) == dists[end]
        # one ELBO estimate per iteration, the selected one keeping its draws
        L = length(optim_trace) - 1
        @test length(online_fit.elbo_estimates) == L
        @test 1 ≤ online_fit.fit_iteration ≤ L
        @test online_fit.fit_distribution == dists[online_fit.fit_iteration + 1]
        for (i, est) in enumerate(online_fit.elbo_estimates)
            @test est isa Pathfinder.ELBOEstimate
            @test length(est.log_density_ratios) == ndraws_elbo
            if i == online_fit.fit_iteration
                @test size(est.draws) == (n, ndraws_elbo)
            else
                @test size(est.draws) == (n, 0)
            end
        end
        values = getproperty.(online_fit.elbo_estimates, :value)
        @test online_fit.fit_iteration == last(Pathfinder._findmax_skipnan(values))
        @test online_fit.best == Pathfinder._findmax_skipnan(values)

        # reproducible for a given rng
        rng = MersenneTwister(11)
        online_fit2 = Pathfinder.OnlineFit(prob, logp; rng, ndraws_elbo, history_length)
        Pathfinder.optimize_with_trace(prob, Optim.LBFGS(); online_fit=online_fit2)
        @test online_fit2.fit_iteration == online_fit.fit_iteration
        @test getproperty.(online_fit2.elbo_estimates, :value) == values
        @test online_fit2.elbo_estimates[online_fit2.fit_iteration].draws ==
            online_fit.elbo_estimates[online_fit.fit_iteration].draws

        @testset "gradient not reported by the optimizer" begin
            # Nelder-Mead reports no gradient, so the fit must compute it itself
            optimizer = Optim.NelderMead()
            rng = MersenneTwister(5)
            online_fit = Pathfinder.OnlineFit(prob, logp; rng, ndraws_elbo, history_length)
            _, optim_trace_online = Pathfinder.optimize_with_trace(
                prob, optimizer; online_fit, maxiters=20
            )
            _, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer; maxiters=20)
            @test optim_trace_online.log_densities == optim_trace.log_densities
            @test online_fit.lbfgs_state.θ ≈ optim_trace.points[end]
            @test online_fit.lbfgs_state.∇logpθ ≈ optim_trace.gradients[end]

            # ... which needs a gradient function on the problem
            ℓ0 = build_logdensityproblem(f, n, 0)
            fun0 = Pathfinder.build_optim_function(
                ℓ0, ADTypes.AutoForwardDiff(), LogDensityProblems.capabilities(ℓ0)
            )
            prob0 = SciMLBase.OptimizationProblem(fun0, x0)
            online_fit = Pathfinder.OnlineFit(prob0, logp; rng, ndraws_elbo, history_length)
            @test_throws ArgumentError Pathfinder.optimize_with_trace(
                prob0, optimizer; online_fit, maxiters=20
            )
        end
    end

    @testset "_gradient_function" begin
        g(x) = -x .^ 3
        p = nothing
        fun_oop = SciMLBase.OptimizationFunction{false}((x, p) -> 0.0; grad=(x, p) -> -g(x))
        fun_iip = SciMLBase.OptimizationFunction{true}(
            (x, p) -> 0.0; grad=(res, x, p) -> (res .= -g(x))
        )
        fun_none = SciMLBase.OptimizationFunction{true}((x, p) -> 0.0)
        x = randn(4)
        @test Pathfinder._gradient_function(fun_oop, p)(x) ≈ g(x)
        @test Pathfinder._gradient_function(fun_iip, p)(x) ≈ g(x)
        @test Pathfinder._gradient_function(fun_none, p) === nothing
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
        # First callback progress depends on backend iteration numbering
        # (0-indexed in Optim v1, 1-indexed in Optim v2)
        @test logs[2].kwargs[:progress] isa Number
        @test 0.0 <= logs[2].kwargs[:progress] <= 0.001
        @test logs[2].message == "Optimizing"
        @test logs[3].kwargs[:progress] > logs[2].kwargs[:progress]
        @test logs[end].kwargs[:progress] == "done"
    end
end
