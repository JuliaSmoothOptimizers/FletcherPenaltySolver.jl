#=
Add tolerances for the subproblem
  -
  -
  -
  -
Add how we decrease the tolerance as a function of sigma?
  - atol
  - rtol
Threshold for restoration
  - stalling
  - unbounded
  - unsuccessful

Random:
  - set the radius of the ball ? (function of atol and sigma)
=#
struct AlgoData{T <: Real}

  #Initialize, Update and Bound parameters of the penalized problem:
  σ_0::T
  σ_max::T
  σ_update::T
  ρ_0::T
  ρ_max::T
  ρ_update::T
  δ_0::T

  #Bound on the Lagrange multipliers
  yM::T

  #Algorithmic parameters
  Δ::T #expected decrease in feasibility between two iterations

  #Functions used in the algorithm
  linear_system_solver::Function

  unconstrained_solver::Function
  atol_sub::Function # (stp.meta.atol)
  rtol_sub::Function #(stp.meta.rtol)

  hessian_approx
  convex_subproblem::Bool #Useful to set the `convex` option in Knitro
end

function AlgoData(
  T::DataType;
  σ_0::Real = one(T),
  σ_max::Real = 1 / eps(T),
  σ_update::Real = T(1.5),
  ρ_0::Real = one(T),
  ρ_max::Real = 1 / eps(T),
  ρ_update::Real = T(1.5),
  δ_0::Real = √eps(T),
  yM::Real = typemax(T),
  Δ::Real = T(0.95),
  linear_system_solver::Function = _solve_ldlt_factorization, #_solve_with_linear_operator,
  unconstrained_solver::Function = is_knitro_installed ? knitro : ipopt,
  atol_sub::Function = atol -> max(0.1, atol),
  rtol_sub::Function = rtol -> max(0.1, rtol),
  hessian_approx = Val(2),
  convex_subproblem::Bool = false,
  kwargs...,
)
  return AlgoData(
    σ_0,
    σ_max,
    σ_update,
    ρ_0,
    ρ_max,
    ρ_update,
    δ_0,
    yM,
    Δ,
    linear_system_solver,
    unconstrained_solver,
    atol_sub,
    rtol_sub,
    hessian_approx,
    convex_subproblem,
  )
end

AlgoData(; kwargs...) = AlgoData(Float64; kwargs...)

abstract type UnconstrainedSolver end

mutable struct KnitroSolver end
mutable struct IpoptSolver end
mutable struct LBFGSSolver end

mutable struct FPSSSolver{T <: Real, QDS <: QDSolver, US <: UnconstrainedSolver}
  meta::AlgoData{T} # AlgoData
  workspace # allocated space for the solver itself
  qdsolver::QDS # solver structure for the linear algebra part, contains allocation for this par
  unconstrained_solver::US # should be a structure/named typle, with everything related to unconstrained
end

#=
function LBFGSSolver{T, V}(
  meta::AbstractNLPModelMeta;
) where {T, V}
  nvar = meta.nvar
  workspace = (
    x = V(undef, nvar),
    xt = V(undef, nvar),
    gx = V(undef, nvar),
    gt = V(undef, nvar),
    d = V(undef, nvar),
  )
  return LBFGSSolver{T, V}(workspace)
end
=#
