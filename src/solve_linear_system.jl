function solve_two_least_squares(
  nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS1, SS2}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P, SS1, SS2}
  # rhs1 and rhs2 are both of size nlp.meta.nvar
  #=
  We solve || ∇c' q - rhs || + δ || q ||^2
  =#
  Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  (q1, stats1) = lsqr!(
    nlp.qdsolver.solver_struct_least_square,
    Aop',
    rhs1,
    λ = √nlp.δ,
    atol = nlp.qdsolver.ls_atol,
    rtol = nlp.qdsolver.ls_rtol,
    itmax = nlp.qdsolver.ls_itmax,
  )
  # nlp.qdsolver.q1 .= q1
  nlp.qdsolver.p1 .= rhs1 - Aop' * q1
  if !stats1.solved
    @warn "Failed solving 1st linear system."
  end

  (q2, stats2) = lsqr!(
    nlp.qdsolver.solver_struct_least_square,
    Aop',
    rhs2,
    λ = √nlp.δ,
    atol = nlp.qdsolver.ls_atol,
    rtol = nlp.qdsolver.ls_rtol,
    itmax = nlp.qdsolver.ls_itmax,
  )
  # nlp.qdsolver.q2 .= q2
  nlp.qdsolver.p2 .= rhs2 - Aop' * q2
  if !stats2.solved
    @warn "Failed solving 2nd linear system."
  end

  return nlp.qdsolver.p1, q1, nlp.qdsolver.p2, q2
end

function solve_two_mixed(
  nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS1, SS2}},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P, SS1, SS2}
  # rhs1 is of size nlp.meta.nvar
  # rhs2 is of size nlp.meta.ncon
  #=
  We solve || ∇c' q - rhs || + δ || q ||^2
  =#
  Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  (q1, stats1) = lsqr!(
    nlp.qdsolver.solver_struct_least_square,
    Aop',
    rhs1,
    λ = √nlp.δ,
    atol = nlp.qdsolver.ls_atol,
    rtol = nlp.qdsolver.ls_rtol,
    itmax = nlp.qdsolver.ls_itmax,
  )
  # nlp.qdsolver.q1 .= q1
  nlp.qdsolver.p1 .= rhs1 - Aop' * q1
  if !stats1.solved
    @warn "Failed solving 1st linear system."
  end

  if nlp.δ != 0.0
    (p2, q2, stats2) = craig!(
      nlp.qdsolver.solver_struct_least_norm,
      Aop,
      -rhs2,
      M = 1 / nlp.δ * opEye(nlp.nlp.meta.ncon),
      sqd = true,
      atol = nlp.qdsolver.ln_atol,
      rtol = nlp.qdsolver.ln_rtol,
      btol = nlp.qdsolver.ln_btol,
      conlim = nlp.qdsolver.ln_conlim,
      itmax = nlp.qdsolver.ln_itmax,
    )
  else
    (p2, q2, stats2) = craig!(
      nlp.qdsolver.solver_struct_least_norm, 
      Aop, 
      -rhs2,
      atol = nlp.qdsolver.ln_atol,
      rtol = nlp.qdsolver.ln_rtol,
      btol = nlp.qdsolver.ln_btol,
      conlim = nlp.qdsolver.ln_conlim,
      itmax = nlp.qdsolver.ln_itmax,
    )
  end
  nlp.qdsolver.p2 .= -p2
  # nlp.qdsolver.q2 .= q2

  if !stats2.solved
    @warn "Failed solving 2nd linear system."
  end

  return nlp.qdsolver.p1, q1, nlp.qdsolver.p2, q2
end

function solve_two_least_squares(
  nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P}
  #set the memory for the matrix in the FletcherPenaltyNLP
  nnzj = nlp.nlp.meta.nnzj
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  # nnz = nvar + nnzj + ncon
  rows = nlp.qdsolver.rows # zeros(Int, nnz)
  cols = nlp.qdsolver.cols # zeros(Int, nnz)
  vals = nlp.qdsolver.vals # zeros(T, nnz)

  #=
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
  nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver},
  x::AbstractVector{T},
  rhs1::AbstractVector,
  rhs2::AbstractVector,
) where {T, S, Tt, A, P}
  # set the memory for the matrix in the FletcherPenaltyNLP
  nnzj = nlp.nlp.meta.nnzj
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  # nnz = nvar + nnzj + ncon
  rows = nlp.qdsolver.rows # zeros(Int, nnz)
  cols = nlp.qdsolver.cols # zeros(Int, nnz)
  vals = nlp.qdsolver.vals # zeros(T, nnz)

  # J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = nvar .+ (1:nnzj)
  @views jac_coord!(nlp.nlp, x, vals[nnz_idx])
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nvar .+ nnzj .+ (1:ncon)
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
