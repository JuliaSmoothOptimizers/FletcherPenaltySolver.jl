
#=
Parameters for the system
[In A'; A -nlp.δ*Im]
that we are solving twice
=#
abstract type QDSolver end

struct IterativeSolver{T<:AbstractFloat, S, SS<:KrylovSolver{T,S}} <: QDSolver
  solver::Function
  M # =opEye(), 
  #λ::T # =zero(T), 
  atol::T # =√eps(T), 
  rtol::T # =√eps(T),
  #radius :: T=zero(T), 
  itmax::Integer # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false

  #allocations
  solver_struct::SS
  # allocation of the linear operator, only one as the matrix is symmetric
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
  m,
  n,
  ::T;
  solver=cg,
  M = opEye(),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  itmax::Integer = 5 * (m + n),
  solver_struct::KrylovSolver{T,Vector{T}}=LsqrSolver(zeros(T, n, m), zeros(T,n)),
) where {T}
  return IterativeSolver(
    solver,
    M,
    atol,
    rtol,
    itmax,
    solver_struct,
    Vector{T}(undef, m + n), 
    Vector{T}(undef, m), 
    Vector{T}(undef, n),
    Vector{T}(undef, n),
    Vector{T}(undef, m),
    Vector{T}(undef, n),
    Vector{T}(undef, m),
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
end
#=
function LDLtSolver(nlp, ::T)
  nnzj = nlp.meta.nnzj
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon

  nnz = nvar + nnzj + ncon
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(T, nnz)
  return LDLtSolver(nnz, rows, cols, vals)
end
=#

struct DirectSolver <: QDSolver end
#=
 - Store the matrix ?
 - in-place LU factorization ?
=#
struct LUSolver <: QDSolver end

const qdsolvers = Dict(
  :Iterative => :_solve_with_linear_operator,
  :LDLt => :_solve_ldlt_factorization,
  #:Direct => :_solve_system_dense,
  #:LU => :_solve_system_factorization_lu,
)

function solve(A, b::AbstractVector, qdsolver::IterativeSolver; kwargs...)
  return qdsolver.solver(
    A, 
    b;
    M = qdsolver.M,
    atol = qdsolver.atol,
    rtol = qdsolver.rtol,
    itmax = qdsolver.itmax, 
    kwargs...,
  )
end

#=
Another solve function for Block systems.
=#
