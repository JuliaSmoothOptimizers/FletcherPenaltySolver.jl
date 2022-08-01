"""
    invJtJJv, invJtJSsv = solve_two_extras(nlp, x, rhs1, rhs2)

The `IterativeSolver` variant solve successively a regularized least square, see [`solve_least_square`](@ref),
and a regularized `minres`. It returns only a warning if the method failed.

The `LDLtSolver` variant use successively a regularized `cgls` and a regularized `minres`.
"""
function solve_two_extras end

"""
    p1, q1, p2, q2 = solve_two_least_squares(nlp, x, rhs1, rhs2)

Solve successively two least square regularized by `√nlp.δ`:
```math
    min || ∇c' q - rhs || + δ || q ||^2
```
`rhs1` and `rhs2` are both of size `nlp.meta.nvar`.

The `IterativeSolver` variant uses two calls to a `Krylov.jl` method, see [`solve_least_square`](@ref).
Note that `nlp.Aop` is not re-evaluated in this case. It returns only a warning if the method failed.

The `LDLtSolver` variant use an LDLt factorization to solve the large system.
"""
function solve_two_least_squares end

"""
    p1, q1, p2, q2 = solve_two_mixed(nlp, x, rhs1, rhs2)

Solve successively a least square regularized by `√nlp.δ`:
```math
    min || ∇c' q - rhs || + δ || q ||^2
```
and a least-norm problem.

`rhs1` is of size `nlp.meta.nvar`, and `rhs2` is of size `nlp.meta.ncon`.

The `IterativeSolver` variant uses two calls to a `Krylov.jl` method, see [`solve_least_square`](@ref) and [`solve_least_norm`](@ref).
It returns only a warning if the method failed.

The `LDLtSolver` variant use an LDLt factorization to solve the large system.
"""
function solve_two_mixed end

function solve_two_extras(
  nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS1, SS2, SS3, It}},
  x::AbstractVector{T},
  rhs1::AbstractVector, # size nvar
  rhs2::AbstractVector, # size ncon
) where {T, S, A, P, SS1, SS2, SS3, It}
  τ = T(max(nlp.δ, 1e-14)) # should be a parameter in the solver structure
  # (invJtJJv, invJtJJvstats) = cgls(nlp.Aop', rhs1, λ = τ)
  (invJtJJv, invJtJJvstats) = solve_least_square(nlp.qdsolver, nlp.Aop' * demiQbound(nlp, x), rhs1, √τ)
  if !invJtJJvstats.solved
    @warn "Failed solving 1st linear system lsqr in extra."
  end

  JtJ = nlp.Aop * Qbound(nlp, x) * nlp.Aop'
  # (invJtJSsv, stats) = minres(JtJ, rhs2, λ = τ)
  solve!(
    nlp.qdsolver.solver_struct_pinv,
    JtJ,
    rhs2,
    λ = τ,
    atol = nlp.qdsolver.ne_atol,
    rtol = nlp.qdsolver.ne_rtol,
    ratol = nlp.qdsolver.ne_ratol,
    rrtol = nlp.qdsolver.ne_rrtol,
    etol = nlp.qdsolver.ne_etol,
    itmax = nlp.qdsolver.ne_itmax,
    conlim = nlp.qdsolver.ne_conlim,
  )
  stats = nlp.qdsolver.solver_struct_pinv.stats
  invJtJSsv = nlp.qdsolver.solver_struct_pinv.x
  if !stats.solved
    @warn "Failed solving 2nd linear system minres in extra."
  end
  return invJtJJv, invJtJSsv
end

function solve_two_least_squares(
  nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS1, SS2, SS3, It}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, A, P, SS1, SS2, SS3, It}
  # We trust this one
  # nlp.Aop .= jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  (q1, stats1) = solve_least_square(nlp.qdsolver, nlp.Aop' * demiQbound(nlp, x), rhs1, √nlp.δ)
  # nlp.qdsolver.q1 .= q1
  # nlp.qdsolver.p1 = rhs1 - nlp.Aop' * q1
  mul!(nlp.qdsolver.p1, nlp.Aop', q1) # nlp.qdsolver.p1 .= nlp.Aop' * q1
  @. nlp.qdsolver.p1 = rhs1 - nlp.qdsolver.p1
  if !stats1.solved
    @warn "Failed solving 1st linear system lsqr."
  end

  (q2, stats2) = solve_least_square(nlp.qdsolver, nlp.Aop' * demiQbound(nlp, x), rhs2, √nlp.δ)
  # nlp.qdsolver.q2 .= q2
  # nlp.qdsolver.p2 = rhs2 - nlp.Aop' * q2
  mul!(nlp.qdsolver.p2, nlp.Aop', q2) # nlp.qdsolver.p2 .= nlp.Aop' * q2
  @. nlp.qdsolver.p2 = rhs2 - nlp.qdsolver.p2
  if !stats2.solved
    @warn "Failed solving 2nd linear system lsqr."
  end
  return nlp.qdsolver.p1, q1, nlp.qdsolver.p2, q2
end

function solve_two_mixed(
  nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS1, SS2, SS3, It}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, A, P, SS1, SS2, SS3, It}
  # rhs1 is of size nlp.meta.nvar
  # rhs2 is of size nlp.meta.ncon
  #=
  We solve || ∇c' q - rhs || + δ || q ||^2
  =#
  nlp.Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  (q1, stats1) = solve_least_square(nlp.qdsolver, nlp.Aop' * semiQbound(nlp, x), semiQbound(nlp, x) * rhs1, √nlp.δ)

  # nlp.qdsolver.q1 .= q1
  mul!(nlp.qdsolver.p1, nlp.Aop', q1) # nlp.qdsolver.p1 .= nlp.Aop' * q1
  @. nlp.qdsolver.p1 = rhs1 - nlp.qdsolver.p1
  if !stats1.solved
    @warn "Failed solving 1st linear system lsqr in mixed."
  end

  (p2, q2, stats2) = solve_least_norm(nlp.qdsolver, semiQbound(nlp, x) * nlp.Aop, -rhs2, nlp.δ)
  @. nlp.qdsolver.p2 = -p2
  # nlp.qdsolver.q2 .= q2

  if !stats2.solved
    @warn "Failed solving 2nd linear system craig in mixed."
  end
  return nlp.qdsolver.p1, q1, nlp.qdsolver.p2, q2
end

function solve_two_extras(
  nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver{Tt, S2, Si, Str}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P, S2, Si, Str}
  τ = T(max(nlp.δ, 1e-14)) # should be a parameter in the solver structure
  nlp.Aop = jac_op(nlp.nlp, x)
  (invJtJJv, invJtJJvstats) = cgls(nlp.Aop', rhs1, λ = τ) # use Krylov.solve!

  JtJ = nlp.Aop * nlp.Aop'
  (invJtJSsv, stats) = minres(JtJ, rhs2, λ = τ) # use Krylov.solve!
  return invJtJJv, invJtJSsv
end

function solve_two_least_squares(
  nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver{Tt, S2, Si, Str}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P, S2, Si, Str}
  #set the memory for the matrix in the FletcherPenaltyNLP
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  # nnz = nvar + nnzj + ncon
  #=
  nnzj = nlp.nlp.meta.nnzj
  rows = nlp.qdsolver.rows # zeros(Int, nnz)
  cols = nlp.qdsolver.cols # zeros(Int, nnz)
  vals = nlp.qdsolver.vals # zeros(T, nnz)

  # WE TRUST THAT two_mixed has been run just before and allocated
  # J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = nvar .+ (1:nnzj)
  @views jac_coord!(nlp.nlp, x, vals[nnz_idx])
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nvar .+ nnzj .+ (1:ncon)
  vals[nnz_idx] .= -nlp.δ

  M = Symmetric(sparse(rows, cols, vals, nvar + ncon, nvar + ncon), :U)
  ldl_factorize!(M, nlp.qdsolver.str)
  =#
  nlp.qdsolver.sol[1:nvar, 1] .= rhs1
  nlp.qdsolver.sol[(nvar + 1):(nvar + ncon), 1] .= 0
  nlp.qdsolver.sol[1:nvar, 2] .= rhs2
  nlp.qdsolver.sol[(nvar + 1):(nvar + ncon), 2] .= 0
  sol = nlp.qdsolver.sol
  if factorized(nlp.qdsolver.str)
    ldiv!(nlp.qdsolver.str, sol)
  else
    @warn "_solve_ldlt_factorization: failed _factorization"
  end

  return sol[1:nvar, 1],
  sol[(nvar + 1):(nvar + ncon), 1],
  sol[1:nvar, 2],
  sol[(nvar + 1):(nvar + ncon), 2]
end

function solve_two_mixed(
  nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver{Tt, S2, Si, Str}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P, S2, Si, Str}
  # set the memory for the matrix in the FletcherPenaltyNLP
  nnzj = nlp.nlp.meta.nnzj
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  # nnz = nvar + nnzj + ncon
  rows = nlp.qdsolver.rows # zeros(Int, nnz)
  cols = nlp.qdsolver.cols # zeros(Int, nnz)
  vals = nlp.qdsolver.vals # zeros(T, nnz)

  # Q^{1/2} * J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = (1 + nvar):(nvar + nnzj)
  @views jac_coord!(nlp.nlp, x, vals[nnz_idx])
  sqb = demiqbound(nlp, x)
  for j=1:nnzj
    vals[nvar + j] = sqb[ nlp.qdsolver.rows[j]] * vals[nvar + j]
  end
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = (1 + nvar + nnzj):(ncon + nvar + nnzj) # nvar .+ nnzj .+ (1:ncon)
  vals[nnz_idx] .= -nlp.δ

  M = Symmetric(sparse(rows, cols, vals, nvar + ncon, nvar + ncon), :U)
  ldl_factorize!(M, nlp.qdsolver.str)

  nlp.qdsolver.sol[1:nvar, 1] .= rhs1
  nlp.qdsolver.sol[(nvar + 1):(nvar + ncon), 1] .= 0
  nlp.qdsolver.sol[1:nvar, 2] .= 0
  nlp.qdsolver.sol[(nvar + 1):(nvar + ncon), 2] .= rhs2
  sol = nlp.qdsolver.sol

  if factorized(nlp.qdsolver.str)
    ldiv!(nlp.qdsolver.str, sol)
  else
    @warn "_solve_ldlt_factorization: failed _factorization"
  end

  return sol[1:nvar, 1],
  sol[(nvar + 1):(nvar + ncon), 1],
  sol[1:nvar, 2],
  sol[(nvar + 1):(nvar + ncon), 2]
end
