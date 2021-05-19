
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
  SS2 <: KrylovSolver{T, S}
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

  #allocations
  solver_struct_least_square::SS1
  solver_struct_least_norm::SS2
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
  solver_struct_least_square::KrylovSolver{T, Vector{T}} = LsqrSolver(
    zeros(T, nlp.meta.nvar, nlp.meta.ncon),
    zeros(T, nlp.meta.nvar),
  ),
  solver_struct_least_norm::KrylovSolver{T, Vector{T}} = CraigSolver(
    zeros(T, nlp.meta.ncon, nlp.meta.nvar),
    zeros(T, nlp.meta.ncon),
  ),
  kwargs...,
) where {T}
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
    solver_struct_least_square,
    solver_struct_least_norm,
    Vector{T}(undef, nlp.meta.ncon + nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
    Vector{T}(undef, nlp.meta.nvar),
    Vector{T}(undef, nlp.meta.ncon),
  )
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
