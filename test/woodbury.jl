using LinearAlgebra
using Pathfinder: WoodburyPDMat
using PDMats
using Test

include("test_utils.jl")

function decompose(W)
    n, m = size(W.B)
    k = min(n, m)
    UA = W.UA
    Q = W.Q
    UC = W.UC
    X = zeros(eltype(UC), n, n)
    X[diagind(X)] .= true
    X[1:k, 1:k] .= UC'
    T = UA' * Q * X
    return T
end

function test_decompositions(W)
    UA = cholesky(W.A).U
    Q, R = qr(UA' \ W.B)
    C = I + R * W.D * R'
    UC = cholesky(Symmetric(C)).U
    @test W.UA ≈ UA
    @test W.Q ≈ Q
    @test W.UC ≈ UC
end

@testset "WoodburyPDMat" begin
    @testset "A $Atype, D $Dtype" for T in (Float64, Float32),
        Atype in (:dense, :diag),
        Dtype in (:dense, :diag),
        n in (5, 10),
        k in (5, 10)

        A = Atype === :diag ? rand_pd_diag_mat(T, n) : rand_pd_mat(T, n)
        B = randn(T, n, k)
        D = Dtype === :diag ? rand_pd_diag_mat(T, k) : rand_pd_mat(T, k)
        W = @inferred WoodburyPDMat{T} WoodburyPDMat(A, B, D)
        Wmat = A + B * D * B'

        @testset "basic" begin
            @test eltype(W) === T
            @test eltype(Wmat) === T
            @test size(W) == (n, n)
            @test size(W, 1) == n
            @test size(W, 2) == n
            @test Matrix(W) ≈ Wmat
            @test W[3, 5] ≈ Wmat[3, 5]
            @test W ≈ Wmat
            test_decompositions(W)
            M = decompose(W)
            @test M * M' ≈ Wmat
            @test WoodburyPDMat(A, B, big.(D)) isa WoodburyPDMat{BigFloat}
            @test Matrix(WoodburyPDMat(A, B, big.(D))) ≈ Wmat
            Wbig = convert(AbstractMatrix{BigFloat}, W)
            @test Wbig isa WoodburyPDMat{BigFloat}
            @test Wbig ≈ Wmat
            test_decompositions(W)
            @test convert(AbstractMatrix{T}, W) === W
        end

        @testset "inv" begin
            invW = @inferred WoodburyPDMat{T} inv(W)
            @test eltype(invW) === T
            invWmat = inv(Matrix(W))
            @test invW isa WoodburyPDMat
            @test invW ≈ invWmat
            test_decompositions(invW)
        end

        @testset "determinant" begin
            @test @inferred(det(W)) ≈ det(Wmat)
            @test @inferred(logdet(W)) ≈ logdet(Wmat)
            @test @inferred(logabsdet(W))[1] ≈ logabsdet(Wmat)[1]
            @test logabsdet(W)[2] == logabsdet(Wmat)[2]
        end

        @testset "diag" begin
            @test @inferred(diag(W)) ≈ diag(Wmat)
        end

        @testset "adjoint/transpose" begin
            @test W' === W
            @test transpose(W) === W
        end

        @testset "lmul!" begin
            x = randn(T, n)
            y = Wmat * x
            @test lmul!(W, x) === x
            @test x ≈ y

            X = randn(T, n, 5)
            Y = Wmat * X
            @test lmul!(W, X) === X
            @test X ≈ Y
        end

        @testset "mul!" begin
            x = randn(T, n)
            y = similar(x)
            @test mul!(y, W, x) === y
            @test y ≈ Wmat * x

            X = randn(T, n, 5)
            Y = similar(X)
            @test mul!(Y, W, X) === Y
            @test Y ≈ Wmat * X
        end

        @testset "*" begin
            @inferred Union{WoodburyPDMat{Float64},Matrix{Float64}} W * 5.0
            @test W * 5.0 isa WoodburyPDMat
            @test W * 5.0 ≈ Wmat * 5
            test_decompositions(W * 5.0)
            test_decompositions(W * 3)
            @test W * -2 isa Matrix
            @test W * -2 ≈ Wmat * -2

            x = randn(T, n)
            @test W * x ≈ Wmat * x

            X = randn(T, n)
            @test W * X ≈ Wmat * X
        end

        @testset "PDMats.dim" begin
            @test PDMats.dim(W) == n
        end

        @testset "unwhiten" begin
            M = decompose(W)

            x = randn(T, n)
            @test @inferred(unwhiten(W, x)) ≈ M * x

            X = randn(T, n, 100)
            @test @inferred(unwhiten(W, X)) ≈ M * X
        end

        @testset "invunwhiten!" begin
            M = decompose(W)

            x = randn(T, n)
            @test @inferred(Pathfinder.invunwhiten!(similar(x), W, x)) ≈ M' \ x

            X = randn(T, n, 100)
            @test @inferred(Pathfinder.invunwhiten!(similar(X), W, X)) ≈ M' \ X
        end

        @testset "PDMats.quad" begin
            x = randn(T, n)
            @test @inferred(quad(W, x)) ≈ dot(x, Wmat, x)

            u = randn(T, n)
            @test quad(W, Pathfinder.invunwhiten!(similar(u), W, u)) ≈ dot(u, u)

            X = randn(T, n, 100)
            @test @inferred(quad(W, X)) ≈ quad(PDMats.PDMat(Symmetric(Wmat)), X)

            U = randn(T, n, 100)
            @test quad(W, Pathfinder.invunwhiten!(similar(U), W, U)) ≈
                vec(sum(abs2, U; dims=1))
        end

        @testset "PDMats.invquad" begin
            x = randn(T, n)
            @test @inferred(invquad(W, x)) ≈ dot(x, inv(Wmat), x)

            u = randn(T, n)
            @test invquad(W, unwhiten(W, u)) ≈ dot(u, u)

            X = randn(T, n, 100)
            @test @inferred(invquad(W, X)) ≈ invquad(PDMats.PDMat(Symmetric(Wmat)), X)

            U = randn(T, n, 100)
            @test invquad(W, unwhiten(W, U)) ≈ vec(sum(abs2, U; dims=1))
        end
    end
end
