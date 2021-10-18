using LinearAlgebra
using Pathfinder: WoodburyPDMat
using PDMats
using Test

function rand_pd_mat(T, n)
    U = qr(randn(T, n, n)).Q
    return Matrix(Symmetric(U * rand_pd_diag_mat(T, n) * U'))
end

rand_pd_diag_mat(T, n) = Diagonal(rand(T, n))

function decompose(W)
    n, k = size(W.B)
    UA = W.UA
    Q = W.Q
    UC = W.UC
    T = UA' * Q * [UC' 0*I; 0*I I(n - k)]
    return T
end

function test_decompositions(W)
    n, k = size(W.B)
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
        Dtype in (:dense, :diag)

        n, k = 10, 5
        A = Atype === :diag ? rand_pd_diag_mat(T, n) : rand_pd_mat(T, n)
        B = randn(T, n, k)
        D = Dtype === :diag ? rand_pd_diag_mat(T, k) : rand_pd_mat(T, k)
        W = @inferred WoodburyPDMat(A, B, D)
        Wmat = A + B * D * B'

        @testset "basic" begin
            @test eltype(W) === T
            @test eltype(Wmat) === T
            @test size(W) == (10, 10)
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
            invW = @inferred inv(W)
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

            X = randn(T, n)
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

        @testset "PDMats.invquad" begin
            x = randn(T, n)
            @test @inferred(invquad(W, x)) ≈ dot(x, inv(Wmat), x)
        end

        @testset "PDMats.invquad" begin
            x = randn(T, n)
            @test @inferred(invquad(W, x)) ≈ dot(x, inv(Wmat), x)
        end

        @testset "unwhiten" begin
            x = randn(T, n)
            M = decompose(W)
            @test @inferred(unwhiten(W, x)) ≈ M * x
        end
    end
end
