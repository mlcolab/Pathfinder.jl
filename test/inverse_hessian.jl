using ForwardDiff
using LinearAlgebra
using Optim
using Pathfinder
using Test

include("test_utils.jl")

function lbfgs_inverse_hessian_explicit(H₀, S, Y)
    B = [H₀ * Y S]
    R = triu(S'Y)
    E = Diagonal(R)
    D = [0*I -inv(R); -inv(R)' R' \ (E + Y' * H₀ * Y)/R]
    return H₀ + B * D * B'
end

@testset "L-BFGS inverse Hessian construction" begin
    @testset "lbfgs_inverse_hessian" begin
        #! format: off
        S = [[1.719573, 3.294037, -2.008877, 3.901275, 0.214324, 0.400382, 0.113598, -1.804262, 0.465563, 2.465748], [-0.445476, -0.514915, 1.069617, -1.505506, -0.036535, -0.11386, -0.029737, 0.579662, -0.118694, -1.726258], [-0.013354, 0.1981, 0.081886, -0.194172, 0.010499, -0.007944, -0.001039, 0.061604, -0.002621, 0.11145], [0.00648, 0.255961, -0.011901, -0.059042, 0.013587, -0.002064, 0.000302, 0.023457, 0.00257, -0.002343], [-0.016408, 0.015005, 0.009654, 0.016735, -0.000245, -0.002825, -0.00106, 0.004272, -0.004578, 0.002275]]
        Y = -[[-2.357935, -3.343312, 4.659008, -7.065433, -0.228045, -0.584164, -0.156837, 2.863289, -0.631753, -7.152021], [0.610851, 0.522617, -2.480668, 2.726559, 0.038874, 0.166123, 0.041055, -0.9199, 0.161064, 5.007094], [0.018312, -0.201064, -0.189911, 0.351657, -0.011171, 0.011591, 0.001434, -0.097763, 0.003557, -0.323265], [-0.008886, -0.25979, 0.027601, 0.106929, -0.014456, 0.003011, -0.000418, -0.037225, -0.003488, 0.006795], [0.022499, -0.015229, -0.02239, -0.030308, 0.00026, 0.004122, 0.001463, -0.00678, 0.006212, -0.006598]]
        #! format: on
        N = length(S[1])
        α = rand(N)
        H₀ = Diagonal(α)

        @test @inferred(Pathfinder.lbfgs_inverse_hessian(H₀, empty(S), empty(Y))) ≈ H₀

        H = Pathfinder.lbfgs_inverse_hessian(H₀, S, Y)
        Hexp = lbfgs_inverse_hessian_explicit(H₀, reduce(hcat, S), reduce(hcat, Y))
        @test H ≈ Hexp
    end

    @testset "lbfgs_inverse_hessians" begin
        n = 10
        history_length = 5
        logp(x) = logp_banana(x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        nocedal_wright_scaling(α, s, y) = fill!(similar(α), dot(y, s) / sum(abs2, y))
        θ₀ = 10 * randn(n)

        optimizer = Optim.LBFGS(;
            m=history_length, linesearch=Optim.LineSearches.MoreThuente()
        )
        θs, logpθs, ∇logpθs = Pathfinder.maximize_with_trace(logp, ∇logp, θ₀, optimizer)

        # run lbfgs_inverse_hessians with the same initialization as Optim.LBFGS
        Hs = Pathfinder.lbfgs_inverse_hessians(
            θs, ∇logpθs; history_length=history_length, Hinit=nocedal_wright_scaling
        )
        ss = diff(θs)
        ps = (Hs .* ∇logpθs)[1:(end - 1)]
        # check that next direction computed from Hessian is the same as the actual
        # direction that was taken
        @test all(≈(1), dot.(ps, ss) ./ norm.(ss) ./ norm.(ps))
    end
end
