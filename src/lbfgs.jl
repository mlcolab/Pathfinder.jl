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

# history storage for L-BFGS and accessor methods
mutable struct LBFGSHistory{T<:Real}
    const position_diffs::Matrix{T}
    const gradient_diffs::Matrix{T}
    const history_perm::Vector{Int}
    history_length::Int
end

function LBFGSHistory{T}(n::Int, history_length::Int) where {T<:Real}
    position_diffs = Matrix{T}(undef, n, history_length + 1)
    gradient_diffs = Matrix{T}(undef, n, history_length + 1)
    history_perm = collect(1:(history_length + 1))
    return LBFGSHistory{T}(position_diffs, gradient_diffs, history_perm, 0)
end

function _history_matrices(history::LBFGSHistory)
    (; position_diffs, gradient_diffs, history_perm, history_length) = history
    history_inds = @view history_perm[(end - history_length + 1):end]
    return @views (position_diffs[:, history_inds], gradient_diffs[:, history_inds])
end

function _propose_history_update!(
    history::LBFGSHistory, position, position_new, gradient, gradient_new
)
    (; history_perm) = history
    queue_ind = first(history_perm)
    history.position_diffs[:, queue_ind] .= position_new .- position
    history.gradient_diffs[:, queue_ind] .= gradient .- gradient_new
    return history
end

function _proposed_history_updates(history::LBFGSHistory)
    (; position_diffs, gradient_diffs, history_perm) = history
    queue_ind = first(history_perm)
    return @views position_diffs[:, queue_ind], gradient_diffs[:, queue_ind]
end

function _accept_history_update!(history::LBFGSHistory)
    (; history_perm) = history
    circshift!(history_perm, -1)
    history.history_length = min(history.history_length + 1, length(history_perm) - 1)
    return history
end

function _has_positive_curvature(pos_diff, grad_diff, ϵ)
    return dot(grad_diff, pos_diff) > ϵ * sum(abs2, grad_diff)
end

# cache for L-BFGS inverse Hessian approximations
struct LBFGSInverseHessianCache{T<:Real}
    diag_invH0::Vector{T}
    B::Matrix{T}
    D::Matrix{T}
end

function LBFGSInverseHessianCache{T}(n::Int, history_length::Int) where {T<:Real}
    diag_invH0 = ones(T, n)
    B = Matrix{T}(undef, n, 2 * history_length)
    D = zeros(T, 2 * history_length, 2 * history_length)
    return LBFGSInverseHessianCache(diag_invH0, B, D)
end

# state for each iteration of L-BFGS optimization
mutable struct LBFGSState{T<:Real,IH<:WoodburyPDMat{T}}
    const x::Vector{T}
    fx::T
    const ∇fx::Vector{T}
    const history::LBFGSHistory{T}
    const cache::LBFGSInverseHessianCache{T}
    invH::IH
    num_bfgs_updates_rejected::Int
end
function LBFGSState(x, fx, ∇fx, history_length::Int)
    T = Base.promote_eltype(x, fx, ∇fx)
    n = length(x)
    history = LBFGSHistory{T}(n, history_length)
    cache = LBFGSInverseHessianCache{T}(n, history_length)
    invH = lbfgs_inverse_hessian!(cache, history) # H₀ = I
    return LBFGSState{T,typeof(invH)}(copy(x), fx, copy(∇fx), history, cache, invH, 0)
end

function _update_state!(state::LBFGSState, x, fx, ∇fx, invH_init!, ϵ)
    _propose_history_update!(state.history, state.x, x, state.∇fx, ∇fx)
    pos_diff, grad_diff = _proposed_history_updates(state.history)

    # only update inverse Hessian if will not destroy positive definiteness
    if _has_positive_curvature(pos_diff, grad_diff, ϵ)
        _accept_history_update!(state.history)
        invH_init!(state.cache.diag_invH0, pos_diff, grad_diff)
        state.invH = lbfgs_inverse_hessian!(state.cache, state.history)
    else
        state.num_bfgs_updates_rejected += 1
    end

    copyto!(state.x, x)
    state.fx = fx
    copyto!(state.∇fx, ∇fx)
    return state
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
    θs, logpθs, ∇logpθs; (invH_init!)=gilbert_invH_init!, history_length=5, ϵ=1e-12
)
    L = length(θs) - 1
    history_length = min(history_length, L)
    state = LBFGSState(first(θs), first(logpθs), first(∇logpθs), history_length)
    invHs = [deepcopy(state.invH)] # trace of invH

    for (θ, logpθ, ∇logpθ) in Iterators.drop(zip(θs, logpθs, ∇logpθs), 1)
        _update_state!(state, θ, logpθ, ∇logpθ, invH_init!, ϵ)
        push!(invHs, deepcopy(state.invH))
    end

    return invHs, state.num_bfgs_updates_rejected
end

"""
    lbfgs_inverse_hessian!(cache::LBFGSInverseHessianCache, history::LBFGSHistory) -> WoodburyPDMat

Compute approximate inverse Hessian initialized from history stored in `cache` and `history`.

`cache` stores the diagonal of the initial inverse Hessian ``H₀^{-1}`` and the matrices
``B₀`` and ``D₀``, which are overwritten here and are used in the construction of the
returned approximate inverse Hessian ``H^{-1}``.

From Theorem 2.2 of [^Byrd1994], the expression for the inverse Hessian ``H^{-1}`` is

```math
\\begin{align}
B &= \\begin{pmatrix}H_0^{-1} Y & S\\end{pmatrix}\\\\
R &= \\operatorname{triu}(S^\\mathrm{T} Y)\\\\
E &= I \\circ R\\\\
D &= \\begin{pmatrix}
    0 & -R^{-1}\\\\
    -R^{-\\mathrm{T}} & R^\\mathrm{-T} (E + Y^\\mathrm{T} H_0 Y ) R^\\mathrm{-1}\\\\
\\end{pmatrix}\\
H^{-1} &= H_0^{-1} + B D B^\\mathrm{T}
\\end{align}
```

[^Byrd1994]: Byrd, R.H., Nocedal, J. & Schnabel, R.B.
             Representations of quasi-Newton matrices and their use in limited memory methods.
             Mathematical Programming 63, 129–156 (1994).
             doi: [10.1007/BF01582063](https://doi.org/10.1007/BF01582063)
"""
function lbfgs_inverse_hessian!(cache::LBFGSInverseHessianCache, history::LBFGSHistory)
    (; B, D, diag_invH0) = cache
    return lbfgs_inverse_hessian!(B, D, diag_invH0, history)
end
function lbfgs_inverse_hessian!(B_cache, D_cache, diag_invH0, history)
    S, Y = _history_matrices(history)
    J = history.history_length
    B = @view B_cache[:, 1:(2J)]
    D = @view D_cache[1:(2J), 1:(2J)]
    invH0 = Diagonal(diag_invH0)
    iszero(J) && return WoodburyPDMat(invH0, B, D)

    @views begin
        B₁ = B[:, 1:J]
        B₂ = B[:, (J + 1):(2J)]
        D₁₁ = D[1:J, 1:J]
        D₁₂ = D[1:J, (J + 1):(2J)]
        D₂₁ = D[(J + 1):(2J), 1:J]
        D₂₂ = D[(J + 1):(2J), (J + 1):(2J)]
    end

    fill!(D₁₁, false)
    mul!(B₁, invH0, Y)
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

    return WoodburyPDMat(invH0, B, D)
end
