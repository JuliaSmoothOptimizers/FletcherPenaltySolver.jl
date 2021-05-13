function _solve_with_linear_operator( # linear_system_solver
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::AbstractVector{T};
  kwargs...,
) where {T <: AbstractFloat}

  #size(A) : nlp.nlp.meta.ncon x nlp.nlp.meta.nvar
  n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
  nn = nlp.nlp.meta.ncon + nlp.nlp.meta.nvar
  Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  # only one as the matrix is symmetric
  function Mp!(r, v) 
    nlp.qdsolver.Jtv .= v[1:n] + Aop' * v[(n + 1):nn]
    nlp.qdsolver.Jv .= Aop * v[1:n] - nlp.δ * v[(n + 1):nn]
    r .= vcat(nlp.qdsolver.Jtv, nlp.qdsolver.Jv) # necessary ?
    return r
  end
  #LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
  prod = @closure v -> Mp!(nlp.qdsolver.opr, v)
  opM = LinearOperator(T, nn, nn, true, true, prod, prod, prod)

  (sol1, stats1) = solve(opM, rhs1, nlp.qdsolver; kwargs...)
  if !stats1.solved
    @warn "Failed solving linear system with $(nlp.qdsolver.solver)."
  end

  (sol2, stats2) = solve(opM, rhs2, nlp.qdsolver; kwargs...)
  if !stats2.solved
    @warn "Failed solving linear system with $(nlp.qdsolver.solver)."
  end

  return sol1, sol2
end

function _solve_with_linear_operator(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::Nothing;
  kwargs...,
) where {T <: AbstractFloat}

  #size(A) : nlp.nlp.meta.ncon x nlp.nlp.meta.nvar
  n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
  nn = nlp.nlp.meta.ncon + nlp.nlp.meta.nvar
  Aop = jac_op!(nlp.nlp, x, nlp.qdsolver.Jv, nlp.qdsolver.Jtv)
  # only one as the matrix is symmetric
  function Mp!(r, v) 
    nlp.qdsolver.Jtv .= v[1:n] + Aop' * v[(n + 1):nn]
    nlp.qdsolver.Jv .= Aop * v[1:n] - nlp.δ * v[(n + 1):nn]
    r .= vcat(nlp.qdsolver.Jtv, nlp.qdsolver.Jv) # necessary ?
    return r
  end
  #LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
  prod = @closure v -> Mp!(nlp.qdsolver.opr, v)
  opM = LinearOperator(T, nn, nn, true, true, prod, prod, prod)

  # I think the result, sol1 is also in nlp.qdsolver.opr ! (but memoize problem)
  (sol1, stats1) = solve(opM, rhs1, nlp.qdsolver; kwargs...)
  if !stats1.solved
    @warn "Failed solving linear system with $(nlp.qdsolver.solver)."
  end

  return sol1
end

function _solve_ldlt_factorization(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::Nothing;
  kwargs...,
) where {T <: AbstractFloat}
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
  S = ldl_analyze(M)
  S.n_d = nvar
  S.tol = √eps(T)
  S.r1 = √eps(T)
  S.r2 = -√eps(T) #regularization
  ldl_factorize!(M, S)
  sol1 = copy(rhs1)
  if factorized(S)
    ldiv!(S, sol1)
  else
    @warn "_solve_ldlt_factorization: failed _factorization"
  end

  return sol1
end

function _solve_ldlt_factorization(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::AbstractVector{T};
  kwargs...,
) where {T <: AbstractFloat}
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
  S = ldl_analyze(M)
  S.n_d = nvar
  S.tol = √eps(T)
  S.r1 = √eps(T)
  S.r2 = -√eps(T) #regularization
  ldl_factorize!(M, S)
  sol = hcat(rhs1, rhs2)
  if factorized(S)
    ldiv!(S, sol)
  else
    @warn "_solve_ldlt_factorization: failed _factorization"
  end

  return sol[:, 1], sol[:, 2]
end

function _solve_system_dense(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::Nothing;
  kwargs...,
) where {T <: AbstractFloat}
  A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  In = diagm(0 => ones(nlp.meta.nvar))
  Im = diagm(0 => ones(nlp.nlp.meta.ncon))
  M = [In A'; A -nlp.δ*Im] #expensive

  sol1 = M \ rhs1

  return sol1
end

function _solve_system_dense(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::AbstractVector{T};
  kwargs...,
) where {T <: AbstractFloat}
  A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  In = diagm(0 => ones(nlp.meta.nvar))
  Im = diagm(0 => ones(nlp.nlp.meta.ncon))
  M = [In A'; A -nlp.δ*Im] #expensive

  sol1 = M \ rhs1
  sol2 = M \ rhs2

  return sol1, sol2
end

function _solve_system_factorization_lu(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::Union{AbstractVector{T}, Nothing};
  kwargs...,
) where {T <: AbstractFloat}
  n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
  A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  In = Matrix{T}(I, n, n) #spdiagm(0 => ones(nlp.meta.nvar)) ?
  Im = Matrix{T}(I, ncon, ncon)
  M = [In A'; A -nlp.δ*Im] #expensive

  LU = lu(M)

  sol1 = LU \ rhs1

  return sol1
end

function _solve_system_factorization_lu(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::AbstractVector{T};
  kwargs...,
) where {T <: AbstractFloat}
  n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
  A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  In = Matrix{T}(I, n, n) #spdiagm(0 => ones(nlp.meta.nvar)) ?
  Im = Matrix{T}(I, ncon, ncon)
  M = [In A'; A -nlp.δ*Im] #expensive

  LU = lu(M)

  sol1 = LU \ rhs1
  sol2 = LU \ rhs2

  return sol1, sol2
end
