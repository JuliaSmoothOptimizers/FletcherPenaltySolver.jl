
#=
Parameters for the system
[In A'; A -nlp.δ*Im]
that we are solving twice
=#
abstract type QDSolver end

struct IterativeSolver{T<:AbstractFloat} <: QDSolver
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
  # allocation of the linear operator, only one as the matrix is symmetric
  opr::Vector{T}
  Jv::Vector{T}
  Jtv::Vector{T}
end
  
function IterativeSolver(
  m,
  n,
  ::T;
  solver=cg,
  M = opEye(),
  # λ::T = zero(T),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  itmax::Integer = 5 * (m + n),
) where {T}
  return IterativeSolver(solver, M, atol, rtol, itmax, Vector{T}(undef, m + n), Vector{T}(undef, m), Vector{T}(undef, n))
end

struct LDLtSolver <: QDSolver end
struct DirectSolver <: QDSolver end
struct LUSolver <: QDSolver end

const qdsolvers = Dict(
  :Iterative => :_solve_with_linear_operator,
  :LDLt => :_solve_ldlt_factorization,
  :Direct => :_solve_system_dense,
  :LU => :_solve_system_factorization_lu,
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
