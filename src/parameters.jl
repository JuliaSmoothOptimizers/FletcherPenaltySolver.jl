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
  δ_max::T
  δ_update::T
  η_1::T
  η_update::T

  #Bound on the Lagrange multipliers
  yM::T

  #Algorithmic parameters
  Δ::T #expected decrease in feasibility between two iterations

  #Functions used in the algorithm
  unconstrained_solver::Function
  subpb_unbounded_threshold
  atol_sub::Function # (stp.meta.atol)
  rtol_sub::Function #(stp.meta.rtol)

  hessian_approx
  convex_subproblem::Bool #Useful to set the `convex` option in Knitro
end

function AlgoData(
  T::DataType;
  σ_0::Real = T(1e3),
  σ_max::Real = 1 / √eps(T),
  σ_update::Real = T(2),
  ρ_0::Real = one(T),
  ρ_max::Real = 1 / √eps(T),
  ρ_update::Real = T(2),
  δ_0::Real = √eps(T),
  δ_max::Real = 1 / √eps(T),
  δ_update::Real = T(10),
  η_1::Real = zero(T),
  η_update::Real = one(T),
  yM::Real = typemax(T),
  Δ::Real = T(0.95),
  unconstrained_solver::Function = StoppingInterface.is_knitro_installed ? NLPModelsKnitro.knitro : ipopt,
  subpb_unbounded_threshold::Real = 1 / √eps(T),
  atol_sub::Function = atol -> atol,
  rtol_sub::Function = rtol -> rtol,
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
    δ_max,
    δ_update,
    η_1,
    η_update,
    yM,
    Δ,
    unconstrained_solver,
    subpb_unbounded_threshold,
    atol_sub,
    rtol_sub,
    hessian_approx,
    convex_subproblem,
  )
end

AlgoData(; kwargs...) = AlgoData(Float64; kwargs...)

abstract type UnconstrainedSolver end

mutable struct KnitroSolver <: UnconstrainedSolver end
mutable struct IpoptSolver <: UnconstrainedSolver end
mutable struct LBFGSSolver <: UnconstrainedSolver end

mutable struct GNSolver
  # Parameters
  η₁ #::AbstractFloat = meta.feas_η₁,
  η₂ #::AbstractFloat = meta.feas_η₂,
  σ₁ #::AbstractFloat = meta.feas_σ₁,
  σ₂ #::AbstractFloat = meta.feas_σ₂,
  Δ₀ #::T = meta.feas_Δ₀,
  bad_steps_lim #::Integer = meta.bad_steps_lim,

  # workspace
  workspace_zp
  workspace_czp
  workspace_Jd
  workspace_Jv
  workspace_Jtv

  # Compute TR-step
  TR_compute_step # ::KrylovSolver{eltype(S), S}
  aggressive_step # ::KrylovSolver{eltype(S), S}
end

function GNSolver(
  x0::S,
  y0::S;
  η₁::AbstractFloat = 1e-3,
  η₂::AbstractFloat = 0.66,
  σ₁::AbstractFloat = 0.25,
  σ₂::AbstractFloat = 2.0,
  Δ0::AbstractFloat = one(eltype(S)),
  bad_steps_lim::Integer = 3,
  TR_compute_step = LsmrSolver(length(y0), length(x0), S),
  aggressive_step = CgSolver(length(x0), length(x0), S),
) where {S}
  n, m = length(x0), length(y0)
  return GNSolver(
    η₁,
    η₂,
    σ₁,
    σ₂,
    Δ0,
    bad_steps_lim,
    S(undef, n),
    S(undef, m),
    S(undef, m),
    S(undef, m),
    S(undef, n),
    TR_compute_step,
    aggressive_step,
  )
end

const qdsolver_correspondence = Dict(:iterative => IterativeSolver, :ldlt => LDLtSolver)

mutable struct FPSSSolver{T <: Real, QDS <: QDSolver, US <: UnconstrainedSolver, FS}
  meta::AlgoData{T} # AlgoData
  workspace # allocated space for the solver itself
  qdsolver::QDS # solver structure for the linear algebra part, contains allocation for this par
  unconstrained_solver::US # should be a structure/named typle, with everything related to unconstrained
  feasibility_solver::FS
end

#Dict(:iterative => IterativeSolver, :ldlt => LDLtSolver)
function FPSSSolver(nlp::AbstractNLPModel, ::T; qds_solver = :ldlt, kwargs...) where {T}
  meta = AlgoData(T; kwargs...)
  workspace = ()
  qdsolver = qdsolver_correspondence[qds_solver](nlp, zero(T); kwargs...)
  unconstrained_solver = KnitroSolver()
  feasibility_solver = GNSolver(nlp.meta.x0, nlp.meta.y0)
  return FPSSSolver(meta, workspace, qdsolver, unconstrained_solver, feasibility_solver)
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
