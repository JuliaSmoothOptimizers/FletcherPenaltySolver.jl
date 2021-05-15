function solve_two_least_squares(
    nlp::FletcherPenaltyNLP{T, S, A, P, IterativeSolver{T, S, SS}},
    x::AbstractVector{T},
    rhs1::AbstractVector,
    rhs2::AbstractVector
) where {T, S, Tt, A, P, SS}
  # rhs1 and rhs2 are both of size nlp.meta.nvar
  #=
  We solve || ∇c' q - rhs || + δ || q ||^2
  =#
  Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  (q1, stats1) = lsqr!(nlp.qdsolver.solver_struct, Aop', rhs1, λ = nlp.δ)
  nlp.qdsolver.p1 .= rhs1 - Aop' * q1
  if !stats1.solved
    @warn "Failed solving 1st linear system with $(nlp.qdsolver.solver)."
  end

  (q2, stats2) = lsqr!(nlp.qdsolver.solver_struct, Aop', rhs2, λ = nlp.δ)
  nlp.qdsolver.p2 .= rhs2 - Aop' * q2
  if !stats2.solved
    @warn "Failed solving 2nd linear system with $(nlp.qdsolver.solver)."
  end

  return nlp.qdsolver.p1, q1, nlp.qdsolver.p2, q2
end

function solve_two_least_squares(
    nlp::FletcherPenaltyNLP{S, Tt, A, P, LDLtSolver},
    x::AbstractVector{T},
    rhs1::AbstractVector,
    rhs2::AbstractVector
) where {T, S, Tt, A, P}
  rhs1 = vcat(rhs1, zeros(T, ncon))
  rhs2 = vcat(rhs2, zeros(T, ncon))
  #set the memory for the matrix in the FletcherPenaltyNLP
  nnzj = nlp.nlp.meta.nnzj
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  nnz = nvar + nnzj + ncon
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(T, nnz)

  # I (1:nvar, 1:nvar)
  nnz_idx = 1:nvar
  rows[nnz_idx], cols[nnz_idx] = 1:nvar, 1:nvar
  vals[nnz_idx] = ones(T, nvar)
  # J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = nvar .+ (1:nnzj)
  @views jac_structure!(nlp.nlp, cols[nnz_idx], rows[nnz_idx]) #transpose
  cols[nnz_idx] .+= nvar
  @views jac_coord!(nlp.nlp, x, vals[nnz_idx])
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nvar .+ nnzj .+ (1:ncon)
  rows[nnz_idx] .= nvar .+ (1:ncon)
  cols[nnz_idx] .= nvar .+ (1:ncon)
  vals[nnz_idx] .= -nlp.δ

  M = Symmetric(sparse(rows, cols, vals, nvar + ncon, nvar + ncon), :U)
  Str = ldl_analyze(M)
  Str.n_d = nvar
  Str.tol = √eps(T)
  Str.r1 = √eps(T)
  Str.r2 = -√eps(T) #regularization
  ldl_factorize!(M, Str)
  sol = hcat(rhs1, rhs2)
  if factorized(Str)
    ldiv!(Str, sol)
  else
    @warn "_solve_ldlt_factorization: failed _factorization"
  end

  return sol[1:nvar, 1], sol[nvar+1:nvar+ncon, 1], sol[1:nvar, 2], sol[nvar+1:nvar+ncon, 2]
end
