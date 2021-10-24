
# eq 4.9
# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function gilbert_init(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

"""
    lbfgs_inverse_hessians(θs, ∇logpθs; Hinit=gilbert_init, history_length=5, ϵ=1e-12) -> Vector{WoodburyPDMat}

From an L-BFGS trajectory and gradients, compute the inverse Hessian approximations at each point.

Given positions `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian
approximations with the provided `history_length`.
"""
function lbfgs_inverse_hessians(θs, ∇logpθs; Hinit=gilbert_init, history_length=5, ϵ=1e-12)
    L = length(θs) - 1
    θ = θs[1]
    ∇logpθ = ∇logpθs[1]

    # allocate caches/containers
    s = similar(θ) # BFGS update, i.e. sₗ = θₗ₊₁ - θₗ = -λ Hₗ ∇logpθₗ
    y = similar(∇logpθ) # cache for yₗ = ∇logpθₗ₊₁ - ∇logpθₗ = Hₗ₊₁ \ s₁ (secant equation)
    S = Vector{typeof(s)}(undef, 0)
    Y = Vector{typeof(y)}(undef, 0)
    α = fill!(similar(θ), true)
    H = lbfgs_inverse_hessian(Diagonal(α), S, Y) # H₀ = I
    Hs = [H]

    for l in 1:L
        s .= θs[l + 1] .- θ
        y .= ∇logpθ .- ∇logpθs[l + 1]
        if dot(y, s) > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            push!(S, copy(s))
            push!(Y, copy(y))

            # initial diagonal estimate of H
            α = Hinit(α, s, y)

            # replace oldest stored s and y with new ones
            if length(S) > history_length
                s = popfirst!(S)
                y = popfirst!(Y)
            end
        else
            @warn "Skipping inverse Hessian update from iteration $l to avoid negative curvature."
        end

        θ = θs[l + 1]
        ∇logpθ = ∇logpθs[l + 1]
        H = lbfgs_inverse_hessian(Diagonal(α), S, Y)
        push!(Hs, H)
    end
    return Hs
end

"""
    lbfgs_inverse_hessian(H₀, S, Y) -> WoodburyPDMat

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
