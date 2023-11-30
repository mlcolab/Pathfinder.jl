using LinearAlgebra
using Pathfinder:
    WoodburyPDMat,
    WoodburyPDFactorization,
    WoodburyPDRightFactor,
    pdfactorize,
    pdunfactorize
using PDMats
using Test

include("test_utils.jl")

function test_factorization(A, B, D, F)
    A′, B′, D′ = pdunfactorize(F)
    @test A′ ≈ A
    @test A′ + B′ * D′ * B′' ≈ A + B * D * B'
end
test_factorization(W::WoodburyPDMat) = test_factorization(W.A, W.B, W.D, W.F)

@testset "Woodbury factorization" begin
    @testset "WoodburyPDRightFactor" begin
        @testset "T=$T, n=$n" for T in (Float64, Float32), n in (5, 10)
            m = 8
            k = min(n, m)
            U = cholesky(rand_pd_mat(T, n)).U
            Q = qr(randn(T, n, k)).Q
            V = cholesky(rand_pd_mat(T, k)).U
            if n > k
                p = max(n - k, 0)
                Rmat = [V zeros(T, k, p); zeros(T, p, k) I(p)] * Q' * U
            else
                Rmat = V * Q' * U
            end
            R = @inferred WoodburyPDRightFactor(U, Q, V)
            @test R isa AbstractMatrix{T}
            @test transpose(transpose(R)) === R
            @test adjoint(adjoint(R)) === R
            @test @inferred(Matrix(R)) ≈ Rmat
            @test @inferred(Matrix(R')) ≈ Rmat'
            @test Matrix{BigFloat}(R) isa Matrix{BigFloat}
            @test @inferred(copy(R)) isa Matrix
            @test copy(R) ≈ Rmat
            Rinv = @inferred inv(R)
            @test Rinv isa Transpose{T,<:WoodburyPDRightFactor{T}}
            @test Matrix(Rinv) ≈ inv(Rmat)
            for i in 1:n, j in 1:k
                @test R[i, j] ≈ Rmat[i, j]
            end
            @testset for (Z, Zmat) in
                         zip([R, R', Rinv, Rinv'], [Rmat, Rmat', inv(Rmat), inv(Rmat')]),
                sz in [(n,), (n, 2)]

                x = randn(T, sz...)
                @test Z * x ≈ Zmat * x
                @test Z' * x ≈ Zmat' * x
                @test Z \ x ≈ Zmat \ x
                @test Z' \ x ≈ Zmat' \ x
                @test mul!(similar(x), Z, x) ≈ Zmat * x
                @test mul!(similar(x), Z', x) ≈ Zmat' * x
                @test lmul!(Z, copy(x)) ≈ Zmat * x
                @test ldiv!(Z, copy(x)) ≈ Zmat \ x
            end
            @test det(R) ≈ det(Rmat)
            @test logabsdet(R)[1] ≈ logabsdet(Rmat)[1]
            @test logabsdet(R)[2] == logabsdet(Rmat)[2]
        end
    end

    @testset "WoodburyPDFactorization" begin
        @testset "T=$T, n=$n" for T in (Float64, Float32), n in (5, 10)
            m = 8
            k = min(n, m)
            U = cholesky(rand_pd_mat(T, n)).U
            Q = qr(randn(T, n, k)).Q
            V = cholesky(rand_pd_mat(T, k)).U
            if n > k
                p = max(n - k, 0)
                Rmat = [V zeros(T, k, p); zeros(T, p, k) I(p)] * Q' * U
            else
                Rmat = V * Q' * U
            end
            Fmat = Rmat' * Rmat
            F = @inferred WoodburyPDFactorization(U, Q, V)
            @test F isa LinearAlgebra.Factorization{T}
            @test transpose(F) === F
            @test adjoint(F) === F
            @test propertynames(F) == (:L, :R)
            @test propertynames(F, true) == (:L, :R, :U, :Q, :V)
            L, R = F
            @test R isa WoodburyPDRightFactor{T}
            @test L isa Transpose{T,<:WoodburyPDRightFactor{T}}
            @test R == F.R
            @test L == F.L == R'
            @test @inferred(Matrix(F)) ≈ Fmat
            @test Matrix{BigFloat}(F) isa Matrix{BigFloat}
            Finv = @inferred inv(F)
            @test Finv isa WoodburyPDFactorization{T}
            @test Matrix(Finv) ≈ inv(Fmat)
            @test sprint(show, "text/plain", F) ==
                (summary(F) * "\nR factor:\n" * sprint(show, "text/plain", F.R))
            @testset for (Z, Zmat) in zip([F, inv(F)], [Fmat, inv(Fmat)]),
                sz in [(n,), (n, 2)]

                x = randn(T, sz...)
                @test lmul!(Z, copy(x)) ≈ Zmat * x
                @test ldiv!(Z, copy(x)) ≈ Zmat \ x
            end
            @test det(F) ≈ det(Fmat)
            @test logabsdet(F)[1] ≈ logabsdet(Fmat)[1]
            @test logabsdet(F)[2] == logabsdet(Fmat)[2]
            @test logdet(F) ≈ logdet(Fmat)
        end
    end

    @testset "pdfactorize" begin
        @testset "A $Atype, D $Dtype eltype $T, n=$n" for T in (Float64, Float32),
            Atype in (:dense, :diag),
            Dtype in (:dense, :diag),
            n in (5, 10)

            m = 8
            A = Atype === :diag ? rand_pd_diag_mat(T, n) : rand_pd_mat(T, n)
            B = randn(T, n, m)
            D = Dtype === :diag ? rand_pd_diag_mat(T, m) : rand_pd_mat(T, m)
            Wmat = A + B * D * B'
            F = @inferred WoodburyPDFactorization{T} pdfactorize(A, B, D)
            test_factorization(A, B, D, F)

            W = WoodburyPDMat(A, B, D)
            @test pdfactorize(W) === W.F
        end
    end

    @testset "pdunfactorize" begin
        @testset "A $Atype, D $Dtype eltype $T" for T in (Float64, Float32),
            Atype in (:dense, :diag),
            Dtype in (:dense, :diag),
            n in (5, 10)

            k = 8
            A = Atype === :diag ? rand_pd_diag_mat(T, n) : rand_pd_mat(T, n)
            B = randn(T, n, k)
            D = Dtype === :diag ? rand_pd_diag_mat(T, k) : rand_pd_mat(T, k)
            Wmat = A + B * D * B'
            F = pdfactorize(A, B, D)
            A′, B′, D′ = @inferred pdunfactorize(F)
            @test A′ ≈ A
            @test A′ + B′ * D′ * B′' ≈ Wmat
        end
    end
end

@testset "WoodburyPDMat" begin
    @testset "A $Atype, D $Dtype eltype $T, n=$n" for T in (Float64, Float32),
        Atype in (:dense, :diag),
        Dtype in (:dense, :diag),
        n in (5, 10)

        m = 8
        A = Atype === :diag ? rand_pd_diag_mat(T, n) : rand_pd_mat(T, n)
        B = randn(T, n, m)
        D = Dtype === :diag ? rand_pd_diag_mat(T, m) : rand_pd_mat(T, m)
        W = @inferred WoodburyPDMat{T} WoodburyPDMat(A, B, D)
        Wmat = A + B * D * B'
        invWmat = inv(Wmat)

        @testset "basic" begin
            @test eltype(W) === T
            @test eltype(Wmat) === T
            @test size(W) == (n, n)
            @test size(W, 1) == n
            @test size(W, 2) == n
            @test Matrix(W) ≈ Wmat
            @test W[3, 5] ≈ Wmat[3, 5]
            @test W ≈ Wmat
            @test WoodburyPDMat(A, B, big.(D)) isa WoodburyPDMat{BigFloat}
            @test Matrix(WoodburyPDMat(A, B, big.(D))) ≈ Wmat

            Wbig = convert(AbstractMatrix{BigFloat}, W)
            @test Wbig isa WoodburyPDMat{BigFloat}
            @test Wbig ≈ Wmat
            @test convert(AbstractMatrix{T}, W) === W
            @test W.F isa WoodburyPDFactorization
            @test Matrix(W.F) ≈ Wmat

            @test convert(PDMats.AbstractPDMat{T}, W) === W
            Wbig2 = convert(PDMats.AbstractPDMat{BigFloat}, W)
            @test Wbig2 isa WoodburyPDMat{BigFloat}
            @test Wbig2 == Wbig

            @test convert(WoodburyPDMat{T}, W) === W
            @test convert(WoodburyPDMat{BigFloat}, W) == Wbig

            test_factorization(W)
        end

        @testset "factorize" begin
            @test factorize(W) === W.F
        end

        @testset "inv" begin
            invW = @inferred inv(W)
            @test eltype(invW) === T
            @test invW isa WoodburyPDMat
            @test invW ≈ invWmat
            @test Matrix(invW.F) ≈ invWmat
            test_factorization(invW)
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

        @testset "+ ::UniformScaling" begin
            c = rand(T) * I
            @test W + c ≈ Wmat + c
            @test c + W ≈ c + Wmat
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

        @testset "ldiv!" begin
            x = randn(T, n)
            y = Wmat \ x
            @test ldiv!(W, x) === x
            @test x ≈ y

            X = randn(T, n, 5)
            Y = Wmat \ X
            @test ldiv!(W, X) === X
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
            test_factorization(W * 5.0)
            test_factorization(W * 3)
            @test W * -2 isa Matrix
            @test W * -2 ≈ Wmat * -2

            x = randn(T, n)
            @test W * x ≈ Wmat * x

            X = randn(T, n, 2)
            @test W * X ≈ Wmat * X
        end

        @testset "\\" begin
            x = randn(T, n)
            @test W \ x ≈ Wmat \ x

            X = randn(T, n, 2)
            @test W \ X ≈ Wmat \ X
        end

        @testset "/" begin
            x = randn(T, n)
            @test x' / W ≈ x' / Wmat

            X = randn(T, 2, n)
            @test X / W ≈ X / Wmat
        end

        @testset "PDMats.dim" begin
            @test PDMats.dim(W) == n
        end

        @testset "whiten/whiten!" begin
            L, _ = factorize(W)

            x = randn(T, n)
            z = @inferred whiten(W, x)
            @test size(x) == size(x)
            @test dot(z, z) ≈ dot(x, invWmat, x)
            z2 = similar(z)
            @test whiten!(z2, W, x) === z2
            @test z2 ≈ z

            X = randn(T, n, 10)
            Z = @inferred whiten(W, X)
            @test size(Z) == size(X)
            for (x, z) in zip(eachcol(X), eachcol(Z))
                @test dot(z, z) ≈ dot(x, invWmat, x)
            end
            Z2 = similar(Z)
            @test whiten!(Z2, W, X) === Z2
            @test Z2 ≈ Z
        end

        @testset "unwhiten/unwhiten!" begin
            L, _ = factorize(W)

            z = randn(T, n)
            x = @inferred unwhiten(W, z)
            @test size(x) == size(z)
            @test dot(x, invWmat, x) ≈ dot(z, z)
            @test whiten(W, x) ≈ z
            x2 = similar(x)
            @test unwhiten!(x2, W, z) === x2
            @test x2 ≈ x

            Z = randn(T, n, 10)
            X = @inferred unwhiten(W, Z)
            @test whiten(W, X) ≈ Z
            @test size(X) == size(Z)
            for (x, z) in zip(eachcol(X), eachcol(Z))
                @test dot(x, invWmat, x) ≈ dot(z, z)
            end
            X2 = similar(X)
            @test unwhiten!(X2, W, Z) === X2
            @test X2 ≈ X
        end

        @testset "invunwhiten!" begin
            _, R = factorize(W)

            x = randn(T, n)
            @test @inferred(Pathfinder.invunwhiten!(similar(x), W, x)) ≈ R \ x

            X = randn(T, n, 100)
            @test @inferred(Pathfinder.invunwhiten!(similar(X), W, X)) ≈ R \ X
        end

        @testset "quad/quad!" begin
            x = randn(T, n)
            @test @inferred(quad(W, x)) ≈ dot(x, Wmat, x)

            u = randn(T, n)
            @test quad(W, Pathfinder.invunwhiten!(similar(u), W, u)) ≈ dot(u, u)

            X = randn(T, n, 10)
            quad_W_X = @inferred quad(W, X)
            @test quad_W_X ≈ quad(PDMats.PDMat(Symmetric(Wmat)), X)
            quad_W_X2 = similar(quad_W_X)
            @test quad!(quad_W_X2, W, X) === quad_W_X2
            @test quad_W_X2 ≈ quad_W_X

            U = randn(T, n, 10)
            @test quad(W, Pathfinder.invunwhiten!(similar(U), W, U)) ≈
                vec(sum(abs2, U; dims=1))
        end

        @testset "invquad/invquad!" begin
            x = randn(T, n)
            @test @inferred(invquad(W, x)) ≈ dot(x, inv(Wmat), x)

            u = randn(T, n)
            @test invquad(W, unwhiten(W, u)) ≈ dot(u, u)

            X = randn(T, n, 10)
            quad_invW_X = @inferred invquad(W, X)
            @test quad_invW_X ≈ invquad(PDMats.PDMat(Symmetric(Wmat)), X)
            quad_invW_X2 = similar(quad_invW_X)
            @test invquad!(quad_invW_X2, W, X) === quad_invW_X2
            @test quad_invW_X2 ≈ quad_invW_X

            U = randn(T, n, 10)
            @test invquad(W, unwhiten(W, U)) ≈ vec(sum(abs2, U; dims=1))
        end
    end
end
