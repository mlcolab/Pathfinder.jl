
# eq 4.9
# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function gilbert_invH_init!(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    @. α = b / (a / α + y^2 - (a / c) * (s / α)^2)
    return α
end

"""
    lbfgs_inverse_hessians(
        θs, ∇logpθs; Hinit=gilbert_init, history_length=5, ϵ=1e-12
    ) -> Tuple{Vector{WoodburyPDMat},Int}

From an L-BFGS trajectory and gradients, compute the inverse Hessian approximations at each point.

Given positions `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian
approximations with the provided `history_length`.

The 2nd returned value is the number of BFGS updates to the inverse Hessian matrices that
were rejected due to keeping the inverse Hessian positive definite.
"""
function lbfgs_inverse_hessians(
    θs, ∇logpθs; (invH_init!)=gilbert_invH_init!, history_length=5, ϵ=1e-12
)
    L = length(θs) - 1
    θ = θs[1]
    ∇logpθ = ∇logpθs[1]
    n = length(θ)

    # allocate caches/containers
    history_ind = 0 # index of last set history entry
    history_length_min = min(history_length, L)
    history_length_effective = 0 # length of history so far
    s = similar(θ) # cache for BFGS update, i.e. sₗ = θₗ₊₁ - θₗ = -λ Hₗ ∇logpθₗ
    y = similar(∇logpθ) # cache for yₗ = ∇logpθₗ₊₁ - ∇logpθₗ = Hₗ₊₁ \ s₁ (secant equation)
    S = similar(s, n, history_length_min) # history of s
    Y = similar(y, n, history_length_min) # history of y
    α = fill!(similar(θ), true) # diag(H₀)
    H0 = Diagonal(α)
    B0 = similar(α, n, 2 * history_length_min)
    D0 = similar(α, 2 * history_length_min, 2 * history_length_min)

    H = lbfgs_inverse_hessian!(B0, D0, H0, S, Y, history_ind, history_length_effective) # H₀ = I
    Hs = [deepcopy(H)] # trace of H

    num_bfgs_updates_rejected = 0
    for l in 1:L
        θlp1, ∇logpθlp1 = θs[l + 1], ∇logpθs[l + 1]
        s .= θlp1 .- θ
        y .= ∇logpθ .- ∇logpθlp1
        if dot(y, s) > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            # add s and y to history
            history_ind = mod1(history_ind + 1, history_length)
            history_length_effective = max(history_ind, history_length_effective)
            S[1:n, history_ind] .= s
            Y[1:n, history_ind] .= y

            # initial diagonal estimate of H
            invH_init!(α, s, y)
        else
            num_bfgs_updates_rejected += 1
        end

        θ, ∇logpθ = θlp1, ∇logpθlp1
        H = lbfgs_inverse_hessian!(B0, D0, H0, S, Y, history_ind, history_length_effective)
        push!(Hs, deepcopy(H))
    end

    return Hs, num_bfgs_updates_rejected
end

"""
    lbfgs_inverse_hessian!(B₀, D₀, H₀, S₀, Y₀, history_ind, history_length) -> WoodburyPDMat

Compute approximate inverse Hessian initialized from `H₀` from history stored in `S₀` and `Y₀`.

`history_ind` indicates the column in `S₀` and `Y₀` that was most recently added to the
history, while `history_length` indicates the number of first columns in `S₀` and `Y₀`
currently being used for storing history.
`S = S₀[:, history_ind+1:history_length; 1:history_ind]` reorders the columns of `₀` so that the
oldest is first and newest is last.

From Theorem 2.2 of [^Byrd1994], the expression for the inverse Hessian ``H`` is

```math
\\begin{align}
B &= \\begin{pmatrix}H_0 Y & S\\end{pmatrix}\\\\
R &= \\operatorname{triu}(S^\\mathrm{T} Y)\\\\
E &= I \\circ R\\\\
D &= \\begin{pmatrix}
    0 & -R^{-1}\\\\
    -R^{-\\mathrm{T}} & R^\\mathrm{-T} (E + Y^\\mathrm{T} H_0 Y ) R^\\mathrm{-1}\\\\
\\end{pmatrix}\\
H &= H_0 + B D B^\\mathrm{T}
\\end{align}
```

[^Byrd1994]: Byrd, R.H., Nocedal, J. & Schnabel, R.B.
             Representations of quasi-Newton matrices and their use in limited memory methods.
             Mathematical Programming 63, 129–156 (1994).
             doi: [10.1007/BF01582063](https://doi.org/10.1007/BF01582063)
"""
function lbfgs_inverse_hessian!(B0, D0, H₀::Diagonal, S0, Y0, history_ind, history_length)
    J = history_length
    B = @view B0[:, 1:(2J)]
    D = @view D0[1:(2J), 1:(2J)]
    fill!(D, false)
    iszero(J) && return WoodburyPDMat(H₀, B, D)

    hist_inds = [(history_ind + 1):history_length; 1:history_ind]
    @views begin
        S = S0[:, hist_inds]
        Y = Y0[:, hist_inds]
        B₁ = B[:, 1:J]
        B₂ = B[:, (J + 1):(2J)]
        D₁₁ = D[1:J, 1:J]
        D₁₂ = D[1:J, (J + 1):(2J)]
        D₂₁ = D[(J + 1):(2J), 1:J]
        D₂₂ = D[(J + 1):(2J), (J + 1):(2J)]
    end

    mul!(B₁, H₀, Y)
    copyto!(B₂, S)
    mul!(D₂₂, S', Y)
    triu!(D₂₂)
    R = UpperTriangular(D₂₂)
    nRinv = UpperTriangular(D₁₂)
    copyto!(nRinv, -I)
    ldiv!(R, nRinv)
    nRinv′ = LowerTriangular(copyto!(D₂₁, nRinv'))
    tril!(D₂₂) # eliminate all but diagonal
    mul!(D₂₂, Y', B₁, true, true)
    LinearAlgebra.copytri!(D₂₂, 'U', false, false)
    rmul!(D₂₂, nRinv)
    lmul!(nRinv′, D₂₂)

    return WoodburyPDMat(H₀, B, D)
end
