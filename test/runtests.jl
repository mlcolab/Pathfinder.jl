using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Pathfinder
using Test

@testset "Pathfinder.jl" begin
    include("woodbury.jl")
    include("inverse_hessian.jl")

    @testset "fit_mvnormal" begin
        n = 10
        history_length = 5
        # banana distribution
        ϕb(b, x) = [x[1]; x[2] + b * (x[1]^2 - 100); x[3:end]]
        function fb(b, x)
            y = ϕb(b, x)
            Σ = Diagonal([100; ones(length(y) - 1)])
            return -dot(y, inv(Σ), y) / 2
        end
        logp(x) = fb(0.03, x)
        logp(x) = logpdf(d, x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        f(x) = -logp(x)
        g!(y, x) = copyto!(y, -∇logp(x))
        nocedal_wright_scaling(α, s, y) = fill!(similar(α), dot(y, s) / sum(abs2, y))
        θ₀ = 10 * randn(n)

        optimizer = Optim.LBFGS(;
            m=history_length, linesearch=Optim.LineSearches.MoreThuente()
        )
        options = Optim.Options(; store_trace=true, extended_trace=true)
        res = Optim.optimize(f, g!, θ₀, optimizer, options)
        θs = Optim.x_trace(res)
        logpθs = -Optim.f_trace(res)
        ∇logpθs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))

        # run fit_mvnormal with the same initialization as Optim.LBFGS
        dists = Pathfinder.fit_mvnormal(
            θs, ∇logpθs; history_length=history_length, cov_init=nocedal_wright_scaling
        )
        ss = diff(θs)
        Hs = [dist.Σ for dist in dists]
        ps = (Hs .* ∇logpθs)[1:(end - 1)]
        # check that next direction computed from Hessian is the same as the actual
        # direction that was taken
        @test all(≈(1), dot.(ps, ss) ./ norm.(ss) ./ norm.(ps))
    end
end
