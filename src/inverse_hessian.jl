
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
    LBFGSInverseHessianState(θ₀, ∇logpθ₀; Hinit=gilbert_init, history_length=5, ϵ=1e-12)

State of the L-BFGS inverse Hessian reconstruction along a trajectory, advanced one point at
a time with [`update_inverse_hessian!`](@ref) and read with
[`lbfgs_inverse_hessian`](@ref).

[`lbfgs_inverse_hessians`](@ref) folds a stored trajectory through it, and the online fit of
[`optimize_with_trace`](@ref) feeds it one point per optimizer iteration, so both construct
the same inverse Hessian approximations. Its memory is `O(length(θ₀) * history_length)`.
"""
mutable struct LBFGSInverseHessianState{
    T<:Real,
    V<:AbstractVector{T},
    G<:AbstractVector{T},
    MS<:AbstractMatrix{T},
    MY<:AbstractMatrix{T},
    E<:Real,
    F,
}
    θ::V # current point
    ∇logpθ::G # gradient of the log density at `θ`
    const s::V # cache for BFGS update, i.e. sₗ = θₗ₊₁ - θₗ = -λ Hₗ ∇logpθₗ
    const y::G # cache for yₗ = ∇logpθₗ₊₁ - ∇logpθₗ = Hₗ₊₁ \ s₁ (secant equation)
    const S::MS # history of s
    const Y::MY # history of y
    α::V # diag(H₀)
    history_ind::Int # index of last set history entry
    history_length_effective::Int # length of history so far
    num_bfgs_updates_rejected::Int
    const history_length::Int
    const ϵ::E
    const Hinit::F
end

function LBFGSInverseHessianState(
    θ, ∇logpθ; Hinit=gilbert_init, history_length::Int=5, ϵ=1e-12
)
    n = length(θ)
    s = similar(θ)
    y = similar(∇logpθ)
    S = similar(s, n, history_length)
    Y = similar(y, n, history_length)
    α = fill!(similar(θ), true) # H₀ = I
    return LBFGSInverseHessianState(
        θ, ∇logpθ, s, y, S, Y, α, 0, 0, 0, history_length, ϵ, Hinit
    )
end

"""
    update_inverse_hessian!(state::LBFGSInverseHessianState, θ, ∇logpθ) -> state

Advance `state` to the next point `θ` of the trajectory, with gradient `∇logpθ`.

The update to the inverse Hessian is rejected, and counted in
`state.num_bfgs_updates_rejected`, if it would destroy positive definiteness.
"""
function update_inverse_hessian!(state::LBFGSInverseHessianState, θlp1, ∇logpθlp1)
    (; s, y, S, Y, history_length, ϵ) = state
    s .= θlp1 .- state.θ
    y .= state.∇logpθ .- ∇logpθlp1
    if dot(y, s) > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
        # add s and y to history
        history_ind = state.history_ind = mod1(state.history_ind + 1, history_length)
        state.history_length_effective = max(history_ind, state.history_length_effective)
        S[:, history_ind] .= s
        Y[:, history_ind] .= y

        # initial diagonal estimate of H
        state.α = state.Hinit(state.α, s, y)
    else
        state.num_bfgs_updates_rejected += 1
    end

    state.θ, state.∇logpθ = θlp1, ∇logpθlp1
    return state
end

"""
    lbfgs_inverse_hessian(state::LBFGSInverseHessianState) -> WoodburyPDMat

The approximate inverse Hessian at the current point of `state`.
"""
function lbfgs_inverse_hessian(state::LBFGSInverseHessianState)
    (; α, S, Y, history_ind, history_length_effective) = state
    return lbfgs_inverse_hessian(Diagonal(α), S, Y, history_ind, history_length_effective)
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
function lbfgs_inverse_hessians(θs, ∇logpθs; Hinit=gilbert_init, history_length=5, ϵ=1e-12)
    L = length(θs) - 1
    # at most `L` updates are stored, so a shorter history needs no more columns
    state = LBFGSInverseHessianState(
        θs[1], ∇logpθs[1]; Hinit, history_length=min(history_length, L), ϵ
    )
    Hs = [lbfgs_inverse_hessian(state)] # trace of H

    for l in 1:L
        update_inverse_hessian!(state, θs[l + 1], ∇logpθs[l + 1])
        push!(Hs, lbfgs_inverse_hessian(state))
    end

    return Hs, state.num_bfgs_updates_rejected
end

"""
    lbfgs_inverse_hessian(H₀, S₀, Y₀, history_ind, history_length) -> WoodburyPDMat

Compute approximate inverse Hessian initialized from `H₀` from history stored in `S₀` and `Y₀`.

`history_ind` indicates the column in `S₀` and `Y₀` that was most recently added to the
history, while `history_length` indicates the number of first columns in `S₀` and `Y₀`
currently being used for storing history.
`S = S₀[:, history_ind+1:history_length; 1:history_ind]` reorders the columns of `S₀` so
that the oldest is first and newest is last.

From [Byrd1994; Theorem 2.2](@citet), the expression for the inverse Hessian ``H`` is

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

# References

- [Byrd1994](@cite): Byrd et al. Math. Program. 63, 1994.
"""
function lbfgs_inverse_hessian(H₀::Diagonal, S0, Y0, history_ind, history_length)
    J = history_length
    α = H₀.diag
    B = similar(α, size(α, 1), 2J)
    D = fill!(similar(α, 2J, 2J), false)
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

    mul!(B₁, Diagonal(α), Y)
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
