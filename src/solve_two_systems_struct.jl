"""
    QDSolver

Abstract structure handling parameters for the system
```math
  [ In   A'      ]    
  [ A  -nlp.δ*Im ]
```
that we are solving twice.

Its implementation should define:
- [`solve_two_extras`](@ref)
- [`solve_two_least_squares`](@ref)
- [`solve_two_mixed`](@ref)
"""
abstract type QDSolver end

"""
    IterativeSolver(nlp::AbstractNLPModel, ::T) <: QDSolver

It uses `Krylov.jl` methods to solve least-squares and least-norm problems.
"""
struct IterativeSolver{
  T <: AbstractFloat,
  S,
  SS1 <: KrylovSolver{T, T, S},
  SS2 <: KrylovSolver{T, T, S},
  SS3 <: KrylovSolver{T, T, S},
  It <: Integer,
} <: QDSolver
  # parameters for least-square solve
  # ls_M # =opEye(), 
  #λ::T # =zero(T), 
  ls_atol::T # =√eps(T), 
  ls_rtol::T # =√eps(T),
  #radius :: T=zero(T), 
  ls_itmax::It # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false

  # parameters for least-norm solve
  # ln_N # =opEye(), 
  # λ::T # =zero(T), 
  ln_atol::T # =√eps(T), 
  ln_rtol::T # =√eps(T),
  ln_btol::T # =√eps(T),
  ln_conlim::T # =1/√eps(T)
  ln_itmax::It # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false

  # parameters for Jt * J solve
  # ne_M = opEye()
  ne_atol::T # = √eps(T)/100,
  ne_rtol::T # = √eps(T)/100,
  ne_ratol::T # = zero(T),
  ne_rrtol::T # = zero(T),
  ne_etol::T # = √eps(T),
  ne_itmax::It # = 0,
  ne_conlim::T # = 1 / √eps(T),

  #allocations
  solver_struct_least_square::SS1
  solver_struct_least_norm::SS2
  solver_struct_pinv::SS3
  # allocation of the linear operator, only one as the matrix is symmetric
  # ToDo: Il faudrait faire le tri ici car tout n'est pas utilisé
  opr::Vector{T}
  Jv::Vector{T}
  Jtv::Vector{T}
  p1::Vector{T} # part 1: solution of 1st LS
  q1::Vector{T} # part 2: solution of 1st LS
  p2::Vector{T} # part 1: solution of 2nd LS
  q2::Vector{T} # part 2: solution of 2nd LS
end

# import Krylov.LsqrSolver
#=
mutable struct LsqrSolver2{T,S} <: KrylovSolver{T,S}
  x  :: S
  Nv :: S
  w  :: S
  Mu :: S

  function LsqrSolver2(n, m, S, T)
    x  = S(undef, m)
    Nv = S(undef, m)
    w  = S(undef, m)
    Mu = S(undef, n)
    solver = new{T,S}(x, Nv, w, Mu)
    return solver
  end
end
=#

function IterativeSolver(
  nlp::AbstractNLPModel,
  ::T;
  explicit_linear_constraints = false,
  # M = opEye(),
  ls_atol::T = √eps(T),
  ls_rtol::T = √eps(T),
  ls_itmax::Integer = 5 * ((explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon) + nlp.meta.nvar),
  ln_atol::T = √eps(T),
  ln_rtol::T = √eps(T),
  ln_btol::T = √eps(T),
  ln_conlim::T = 1 / √eps(T),
  ln_itmax::Integer = 5 * ((explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon) + nlp.meta.nvar),
  ne_atol::T = √eps(T) / 100,
  ne_rtol::T = √eps(T) / 100,
  ne_ratol::T = zero(T),
  ne_rrtol::T = zero(T),
  ne_etol::T = √eps(T),
  ne_itmax::Int = 0,
  ne_conlim::T = 1 / √eps(T),
  solver_struct_least_square::KrylovSolver{T, T, S} = LsqrSolver(
    nlp.meta.nvar,
    explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon,
    Vector{T},
  ),
  solver_struct_least_norm::KrylovSolver{T, T, S} = CraigSolver( # LnlqSolver(
    explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon,
    nlp.meta.nvar,
    Vector{T},
  ),
  solver_struct_pinv::KrylovSolver{T, T, S} = MinresSolver(explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon, nlp.meta.nvar, Vector{T}),
  kwargs...,
) where {T, S}
  ncon = explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon
  return IterativeSolver(
    # M,
    ls_atol,
    ls_rtol,
    ls_itmax,
    ln_atol,
    ln_rtol,
    ln_btol,
    ln_conlim,
    ln_itmax,
    ne_atol,
    ne_rtol,
    ne_ratol,
    ne_rrtol,
    ne_etol,
    ne_itmax,
    ne_conlim,
    solver_struct_least_square,
    solver_struct_least_norm,
    solver_struct_pinv,
    Vector{T}(undef, ncon + nlp.meta.nvar),
    Vector{T}(undef, ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, ncon),
  )
end

"""
    solve_least_square(qdsolver::IterativeSolver, A, b, λ)

Solve least squares problem with regularization λ.
"""
function solve_least_square(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3, It},
  A,
  b,
  λ,
) where {T, S, SS1, SS2, SS3, It}
  solve!(
    qdsolver.solver_struct_least_square,
    A,
    b,
    λ = λ,
    atol = qdsolver.ls_atol,
    rtol = qdsolver.ls_rtol,
    itmax = qdsolver.ls_itmax,
  )
  x = qdsolver.solver_struct_least_square.x
  stats = qdsolver.solver_struct_least_square.stats
  return (x, stats)
end

#=
CLEAN THE PARAMETERS
function solve_least_square(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3},
  A,
  b,
  λ,
) where {T, S, Tt, App, P, SS1 <: LslqSolver, SS2, SS3}
  return lslq!(
    qdsolver.solver_struct_least_square,
    A,
    b,
    λ = λ, # sqd = true then?
    atol = qdsolver.ls_atol,
    rtol = qdsolver.ls_rtol,
    btol :: T=√eps(T),
    etol :: T=√eps(T),
    utol :: T=√eps(T),
    itmax = qdsolver.ls_itmax,
  )
end
=#

function solve_least_norm(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3, It},
  A,
  b,
  δ,
) where {T, S, SS1, SS2 <: CraigSolver, SS3, It}
  if δ != 0
    craig!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      M = 1 / δ * opEye(length(b)),
      sqd = true,
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      btol = qdsolver.ln_btol,
      conlim = qdsolver.ln_conlim,
      itmax = qdsolver.ln_itmax,
    )
  else
    craig!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      btol = qdsolver.ln_btol,
      conlim = qdsolver.ln_conlim,
      itmax = qdsolver.ln_itmax,
    )
  end
  x, y = qdsolver.solver_struct_least_norm.x, qdsolver.solver_struct_least_norm.y
  stats = qdsolver.solver_struct_least_norm.stats
  return (x, y, stats)
end

"""
    solve_least_norm(qdsolver::IterativeSolver, A, b, λ)

Solve least squares problem with regularization δ.
"""
function solve_least_norm(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3, It},
  A,
  b,
  δ,
) where {T, S, SS1, SS2, SS3, It}
  ncon = length(b)
  if δ != 0
    solve!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      M = 1 / δ * opEye(ncon),
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      itmax = qdsolver.ln_itmax,
    )
  else
    solve!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      itmax = qdsolver.ln_itmax,
    )
  end
  x, y = qdsolver.solver_struct_least_norm.x, qdsolver.solver_struct_least_norm.y
  stats = qdsolver.solver_struct_least_norm.stats
  return (x, y, stats)
end

#=
nnzj = nlp.nlp.meta.nnzj
  nvar, ncon = nlp.nlp.meta.nvar, nlp.nlp.meta.ncon

  nnz = nvar + nnzj + ncon
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(T, nnz)

+ The LDLFactorizationStruct
=#
"""
    LDLtSolver(nlp::AbstractNLPModel, ::T) <: QDSolver

It uses `LDLFactorization.jl` methods to solve least-squares and least-norm problems.
"""
struct LDLtSolver{S, S2, Si, Str} <: QDSolver
  nnz
  rows::Si
  cols::Si
  vals::S
  str::Str # LDLFactorization{T <: Real, Ti <: Integer, Tn <: Integer, Tp <: Integer}
  sol::S2
end

function LDLtSolver(
  nlp,
  ::T;
  explicit_linear_constraints = false,
  ldlt_tol = √eps(T),
  ldlt_r1 = √eps(T),
  ldlt_r2 = -√eps(T),
  kwargs...,
) where {T <: Number}
  nnzj = explicit_linear_constraints ? nlp.meta.nln_nnzj : nlp.meta.nnzj
  nvar = nlp.meta.nvar
  ncon = explicit_linear_constraints ? nlp.meta.nnln : nlp.meta.ncon

  nnz = nvar + nnzj + ncon
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(T, nnz)

  # I (1:nvar, 1:nvar)
  nnz_idx = 1:nvar
  rows[nnz_idx], cols[nnz_idx] = 1:nvar, 1:nvar
  vals[nnz_idx] .= ones(T, nvar)
  # J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = nvar .+ (1:nnzj)
  if explicit_linear_constraints
    @views jac_nln_structure!(nlp, cols[nnz_idx], rows[nnz_idx]) #transpose
  else
    @views jac_structure!(nlp, cols[nnz_idx], rows[nnz_idx]) #transpose
  end
  cols[nnz_idx] .+= nvar
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nvar .+ nnzj .+ (1:ncon)
  rows[nnz_idx] .= nvar .+ (1:ncon)
  cols[nnz_idx] .= nvar .+ (1:ncon)

  M = Symmetric(sparse(rows, cols, vals, nvar + ncon, nvar + ncon), :U)
  Str = ldl_analyze(M)
  Str.n_d = nvar
  Str.tol = ldlt_tol
  Str.r1 = ldlt_r1
  Str.r2 = ldlt_r2 #regularization < 0

  sol = zeros(T, nvar + ncon, 2) # the store the 2 rhs

  return LDLtSolver(nnz, rows, cols, vals, Str, sol)
end

"""
    DirectSolver(nlp::AbstractNLPModel, ::T) <: QDSolver
"""
struct DirectSolver <: QDSolver end
#=
 - Store the matrix ?
 - in-place LU factorization ?
=#
"""
    LUSolver(nlp::AbstractNLPModel, ::T) <: QDSolver
"""
struct LUSolver <: QDSolver end

#=
Another solve function for Block systems.
=#
