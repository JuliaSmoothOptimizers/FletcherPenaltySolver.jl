
#=
Parameters for the system
[In A'; A -nlp.δ*Im]
that we are solving twice
=#
abstract type QDSolver end

struct IterativeSolver{
  T <: AbstractFloat,
  S,
  SS1 <: KrylovSolver{T, S},
  SS2 <: KrylovSolver{T, S},
  SS3 <: KrylovSolver{T, S},
} <: QDSolver
  # parameters for least-square solve
  # ls_M # =opEye(), 
  #λ::T # =zero(T), 
  ls_atol::T # =√eps(T), 
  ls_rtol::T # =√eps(T),
  #radius :: T=zero(T), 
  ls_itmax::Integer # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false

  # parameters for least-norm solve
  # ln_N # =opEye(), 
  # λ::T # =zero(T), 
  ln_atol::T # =√eps(T), 
  ln_rtol::T # =√eps(T),
  ln_btol::T # =√eps(T),
  ln_conlim::T # =1/√eps(T)
  ln_itmax::Integer # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false

  # parameters for Jt * J solve
  # ne_M = opEye()
  ne_atol::T # = √eps(T)/100,
  ne_rtol::T # = √eps(T)/100,
  ne_ratol::T # = zero(T),
  ne_rrtol::T # = zero(T),
  ne_etol::T # = √eps(T),
  ne_itmax::Integer # = 0,
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
  # M = opEye(),
  ls_atol::T = √eps(T),
  ls_rtol::T = √eps(T),
  ls_itmax::Integer = 5 * (nlp.meta.ncon + nlp.meta.nvar),
  ln_atol::T = √eps(T),
  ln_rtol::T = √eps(T),
  ln_btol::T = √eps(T),
  ln_conlim::T = 1 / √eps(T),
  ln_itmax::Integer = 5 * (nlp.meta.ncon + nlp.meta.nvar),
  ne_atol::T = √eps(T) / 100,
  ne_rtol::T = √eps(T) / 100,
  ne_ratol::T = zero(T),
  ne_rrtol::T = zero(T),
  ne_etol::T = √eps(T),
  ne_itmax::Int = 0,
  ne_conlim::T = 1 / √eps(T),
  solver_struct_least_square::KrylovSolver{T, S} = LsqrSolver(
    nlp.meta.nvar,
    nlp.meta.ncon,
    Vector{T},
  ),
  solver_struct_least_norm::KrylovSolver{T, S} = LnlqSolver( # CraigSolver(
    nlp.meta.ncon,
    nlp.meta.nvar,
    Vector{T},
  ),
  solver_struct_pinv::KrylovSolver{T, S} = MinresSolver(nlp.meta.ncon, nlp.meta.nvar, Vector{T}),
  kwargs...,
) where {T, S}
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
    Vector{T}(undef, nlp.meta.ncon + nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
  )
end

function solve_least_square(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3},
  A,
  b,
  λ,
) where {T, S, SS1 <: LsqrSolver, SS2, SS3}
  return lsqr!(
    qdsolver.solver_struct_least_square,
    A,
    b,
    λ = λ,
    atol = qdsolver.ls_atol,
    rtol = qdsolver.ls_rtol,
    itmax = qdsolver.ls_itmax,
  )
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
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3},
  A,
  b,
  δ,
) where {T, S, SS1, SS2 <: CraigSolver, SS3}
  return if δ != 0
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
end

function solve_least_norm(
  qdsolver::IterativeSolver{T, S, SS1, SS2, SS3},
  A,
  b,
  δ,
) where {T, S, SS1, SS2 <: LnlqSolver, SS3}
  return if δ != 0
    lnlq!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      M = 1 / δ * opEye(nlp.nlp.meta.ncon),
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      itmax = qdsolver.ln_itmax,
    )
  else
    lnlq!(
      qdsolver.solver_struct_least_norm,
      A,
      b,
      atol = qdsolver.ln_atol,
      rtol = qdsolver.ln_rtol,
      itmax = qdsolver.ln_itmax,
    )
  end
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
struct LDLtSolver <: QDSolver
  nnz
  rows
  cols
  vals
  str # LDLFactorization{T <: Real, Ti <: Integer, Tn <: Integer, Tp <: Integer}
  sol
end

function LDLtSolver(
  nlp,
  ::T;
  ldlt_tol = √eps(T),
  ldlt_r1 = √eps(T),
  ldlt_r2 = -√eps(T),
  kwargs...,
) where {T <: Number}
  nnzj = nlp.meta.nnzj
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon

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
  @views jac_structure!(nlp, cols[nnz_idx], rows[nnz_idx]) #transpose
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

struct DirectSolver <: QDSolver end
#=
 - Store the matrix ?
 - in-place LU factorization ?
=#
struct LUSolver <: QDSolver end

#=
Another solve function for Block systems.
=#
