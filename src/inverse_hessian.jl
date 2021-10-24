"""
    lbfgs_inverse_hessian(H₀, S, Y)

Compute approximate inverse Hessian initialized from `H₀` from history stored in `S` and `Y`.

From Theorem 2.2 of [^Byrd1994], the expression is

```math
\\begin{align}
B &= \\begin{pmatrix}H_0 Y & S\\end{pmatrix}\\\\
R &= \\operatorname{triu}(S^\\mathrm{T} Y)\\\\
E &= I \\circ R\\\\
D &= \\begin{pmatrix}
    0 & -R^{-1}\\\\
    -R^{-\\mathrm{T}} & R^\\mathrm{-T} (E + Y^\\mathrm{T} H₀ Y ) R^\\mathrm{-1}\\\\
H &= H_0 + B D B^\\mathrm{T}
\\end{pmatrix}
\\end{align}
```

[^Byrd1994]: Byrd, R.H., Nocedal, J. & Schnabel, R.B.
             Representations of quasi-Newton matrices and their use in limited memory methods.
             Mathematical Programming 63, 129–156 (1994).
             doi: [10.1007/BF01582063](https://doi.org/10.1007/BF01582063)
"""
function lbfgs_inverse_hessian(H₀::Diagonal, S, Y)
    J = length(S)
    α = H₀.diag
    B = similar(α, size(α, 1), 2J)
    D = fill!(similar(α, 2J, 2J), false)
    iszero(J) && return WoodburyPDMat(H₀, B, D)

    for j in 1:J
        yⱼ = Y[j]
        sⱼ = S[j]
        B[:, j] .= α .* yⱼ
        B[:, J + j] .= sⱼ
        for i in 1:(j - 1)
            D[J + i, J + j] = dot(S[i], yⱼ)
        end
        D[J + j, J + j] = dot(sⱼ, yⱼ)
    end
    R = @views UpperTriangular(D[(J + 1):(2J), (J + 1):(2J)])
    nRinv = @views UpperTriangular(D[1:J, (J + 1):(2J)])
    copyto!(nRinv, -I)
    ldiv!(R, nRinv)
    nRinv′ = @views LowerTriangular(copyto!(D[(J + 1):(2J), 1:J], nRinv'))
    for j in 1:J
        αyⱼ = @views B[:, j]
        for i in 1:(j - 1)
            D[J + i, J + j] = dot(Y[i], αyⱼ)
        end
        D[J + j, J + j] += dot(Y[j], αyⱼ)
    end
    D22 = @view D[(J + 1):(2J), (J + 1):(2J)]
    LinearAlgebra.copytri!(D22, 'U', false, false)
    rmul!(D22, nRinv)
    lmul!(nRinv′, D22)

    return WoodburyPDMat(H₀, B, D)
end
