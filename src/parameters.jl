"""
    AlgoData(; kwargs...) 
    AlgoData(T::DataType; kwargs...)

Structure containing all the parameters used in the [`fps_solve`](@ref) call.
`T` is the datatype used in the algorithm, by default it is `Float64`.
Returns a `AlgoData` structure.

# Arguments
The keyword arguments may include:
- `σ_0::Real = T(1e3)`: Initialize subproblem's parameter σ;
- `σ_max::Real = 1 / √eps(T)`: Maximum value for subproblem's parameter σ;
- `σ_update::Real = T(2)`: Update subproblem's parameter σ;
- `ρ_0::Real = one(T)`: Initialize subproblem's parameter ρ;
- `ρ_max::Real = 1 / √eps(T)`: Maximum value for subproblem's parameter ρ;
- `ρ_update::Real = T(2)`: Update subproblem's parameter ρ;
- `δ_0::Real = √eps(T)`: Initialize subproblem's parameter δ;
- `δ_max::Real = 1 / √eps(T)`: Maximum value for subproblem's parameter δ;
- `δ_update::Real = T(10)`: Update subproblem's parameter δ;
- `η_1::Real = zero(T)`: Initialize subproblem's parameter η;
- `η_update::Real = one(T)`: Update subproblem's parameter η;
- `yM::Real = typemax(T)`: bound on the Lagrange multipliers;
- `Δ::Real = T(0.95)`: expected decrease in feasibility between two iterations;
- `subproblem_solver::Function = ipopt`: solver used for the subproblem;
- `subpb_unbounded_threshold::Real = 1 / √eps(T)`: below the opposite of this value, the subproblem is unbounded;
- `subsolver_max_iter::Int = 20000`: maximum of iteration for the subproblem solver;
- `atol_sub::Function = atol -> atol`: absolute tolerance for the subproblem in function of `atol`;
- `rtol_sub::Function = rtol -> rtol`: relative tolerance for the subproblem in function of `rtol`;
- `hessian_approx = Val(2)`: either `Val(1)` or `Val(2)`, it selects the hessian approximation;
- `convex_subproblem::Bool = false`: true if the subproblem is convex. Useful to set the `convex` option in `knitro`;
- `lagrange_bound::T = 1 / sqrt(eps(T))`: upper-bound on the Lagrange multiplier.

For more details, we refer to the package documentation [fine-tuneFPS.md](https://JuliaSmoothOptimizers.github.io/FletcherPenaltySolver.jl/dev/fine-tuneFPS/). 
"""
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
  Δ::T

  #Functions used in the algorithm
  subproblem_solver::Function
  subpb_unbounded_threshold
  subsolver_max_iter::Int
  atol_sub::Function # (stp.meta.atol)
  rtol_sub::Function #(stp.meta.rtol)

  hessian_approx
  explicit_linear_constraints::Bool
  convex_subproblem::Bool
  lagrange_bound::T
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
  subproblem_solver::Function = T == Float64 ? ipopt : tron,
  subpb_unbounded_threshold::Real = 1 / √eps(T),
  subsolver_max_iter::Int = 20000,
  atol_sub::Function = atol -> atol,
  rtol_sub::Function = rtol -> rtol,
  hessian_approx = Val(2),
  explicit_linear_constraints = false,
  convex_subproblem::Bool = false,
  lagrange_bound::Real = 1 / sqrt(eps(T)),
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
    subproblem_solver,
    subpb_unbounded_threshold,
    subsolver_max_iter,
    atol_sub,
    rtol_sub,
    hessian_approx,
    explicit_linear_constraints,
    convex_subproblem,
    lagrange_bound,
  )
end

AlgoData(; kwargs...) = AlgoData(Float64; kwargs...)

"""
    GNSolver(x, y; kwargs...)

Structure containing all the parameters used in the feasibility step.
`x` is an intial guess, and `y` is an initial guess for the Lagrange multiplier.
Returns a `GNSolver` structure.

# Arguments
The keyword arguments may include:

- `η₁::T=T(1e-3)`: Feasibility step: decrease the trust-region radius when `Ared/Pred < η₁`.
- `η₂::T=T(0.66)`: Feasibility step: increase the trust-region radius when `Ared/Pred > η₂`.
- `σ₁::T=T(0.25)`: Feasibility step: decrease coefficient of the trust-region radius.
- `σ₂::T=T(2.0)`: Feasibility step: increase coefficient of the trust-region radius.
- `Δ₀::T=one(T)`: Feasibility step: initial radius.
- `bad_steps_lim::Integer=3`: Feasibility step: consecutive bad steps before using a second order step.
- `feas_expected_decrease::T=T(0.95)`: Feasibility step: bad steps are when `‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease`.
- `TR_compute_step = LsmrSolver(length(y0), length(x0), S)`: Compute the direction in feasibility step.
- `aggressive_step = CgSolver(length(x0), length(x0), S)`: Compute the direction in feasibility step in agressive mode.
"""
mutable struct GNSolver
  # Parameters
  η₁
  η₂
  σ₁
  σ₂
  Δ₀
  bad_steps_lim
  feas_expected_decrease

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
  Δ₀::AbstractFloat = one(eltype(S)),
  bad_steps_lim::Integer = 3,
  feas_expected_decrease::AbstractFloat = eltype(S)(0.95),
  TR_compute_step = LsmrSolver(length(y0), length(x0), S),
  aggressive_step = CgSolver(length(x0), length(x0), S),
) where {S}
  n, m = length(x0), length(y0)
  return GNSolver(
    η₁,
    η₂,
    σ₁,
    σ₂,
    Δ₀,
    bad_steps_lim,
    feas_expected_decrease,
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

const subproblem_solver_correspondence = Dict(
  :R2 => :R2Solver,
  :lbfgs => :LBFGSSolver,
  :tron => :TronSolver,
  :trunk => :TrunkSolver,
  :ipopt => :IpoptSolver,
  :knitro => :KnitroSolver,
)

"""
    FPSSSolver(nlp::AbstractNLPModel [, x0 = nlp.meta.x0]; kwargs...)
    FPSSSolver(stp::NLPStopping; kwargs...)

Structure regrouping all the structure used during the `fps_solve` call. It returns a `FPSSSolver` structure.

# Arguments
The keyword arguments may include:

- `stp::NLPStopping`: `Stopping` structure for this algorithm workflow;
- `meta::AlgoData{T}`: see [`AlgoData`](@ref);
- `workspace`: allocated space for the solver itself;
- `qdsolver`: solver structure for the linear algebra part, contains allocation for this part. By default a `LDLtSolver`, but an alternative is `IterativeSolver` ;
- `subproblem_solver::AbstractOptimizationSolver`: by default a `subproblem_solver_correspondence[Symbol(meta.subproblem_solver)]`;
- `sub_stats::GenericExecutionStats`: stats structure for the result of `subproblem_solver`;
- `feasibility_solver`: by default a `GNSolver`, see [`GNSolver`](@ref);
- `model::FletcherPenaltyNLP`: subproblem;
- `sub_stp::NLPStopping`: `Stopping` structure for the subproblem.

Note:
- `subproblem_solver` is accessible from the `subproblem_solver_correspondence::Dict`.
- the `qdsolver` is accessible from the dictionary `qdsolver_correspondence`.
"""
mutable struct FPSSSolver{
  T <: Real,
  S,
  P,
  QDS <: QDSolver,
  US <: AbstractOptimizationSolver,
  FS,
  Pb <: AbstractNLPModel{T, S},
  A <: Union{Val{1}, Val{2}},
  Score,
  M,
  SRC,
  MStp,
  LoS,
} <: AbstractOptimizationSolver
  stp::NLPStopping{Pb}
  meta::AlgoData{T}
  workspace
  qdsolver::QDS
  subproblem_solver::US
  sub_stats::GenericExecutionStats{T, S}
  feasibility_solver::FS
  model::FletcherPenaltyNLP{T, S, A, P, QDS, Pb}
  sub_stp::NLPStopping{
    FletcherPenaltyNLP{T, S, A, P, QDS, Pb},
    M,
    SRC,
    NLPAtX{Score, T, S},
    MStp,
    LoS,
  }
end

function FPSSSolver(nlp::AbstractNLPModel{T, S}, x0::S = nlp.meta.x0; kwargs...) where {T, S}
  cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
  #Tanj: how to handle stopping criteria where tol_check depends on the State?
  Fptc(atol, rtol, opt0) =
    rtol * vcat(ones(T, nlp.meta.ncon) .+ norm(cx0, Inf), ones(T, nlp.meta.nvar) .+ norm(gx0, Inf))
  initial_state = NLPAtX(
    x0,
    fill!(S(undef, nlp.meta.ncon), 0),
    S(undef, nlp.meta.ncon + nlp.meta.nvar),
    cx = cx0,
    gx = gx0,
    res = gx0,
  )
  stp = NLPStopping(
    nlp,
    initial_state,
    optimality_check = Fletcher_penalty_optimality_check,
    atol = T(1e-7), # really convert here ?
    rtol = T(1e-7),
    tol_check = Fptc;
    # max_cntrs = Stopping.init_max_counters();
    kwargs...,
  )
  return FPSSSolver(stp; kwargs...)
end

function FPSSSolver(stp::NLPStopping; qds_solver = :ldlt, kwargs...)
  nlp = stp.pb
  atol = stp.meta.atol
  rtol = stp.meta.rtol
  x, y = get_x0(nlp), get_y0(nlp)
  S = typeof(x)
  T = eltype(x)
  meta = AlgoData(T; kwargs...)
  workspace = ()
  qdsolver = qdsolver_correspondence[qds_solver](nlp, zero(T); kwargs...)
  feasibility_solver = GNSolver(x, y)
  model = FletcherPenaltyNLP(
    nlp,
    meta.σ_0,
    meta.ρ_0,
    zero(T),
    meta.hessian_approx,
    explicit_linear_constraints = meta.explicit_linear_constraints,
    qds = qdsolver,
  )
  sub_stats = GenericExecutionStats(model)
  subproblem_solver = eval(subproblem_solver_correspondence[Symbol(meta.subproblem_solver)])(model)
  sub_state = if model.meta.ncon > 0
    NLPAtX(x, model.meta.y0, Jx = jac(model, x), res = fill!(S(undef, nlp.meta.nvar), 0)) # eval Jx
  else
    NLPAtX(x, res = fill!(S(undef, nlp.meta.nvar), 0))
  end
  sub_stp = NLPStopping(
    model,
    sub_state,
    main_stp = stp,
    optimality_check = if model.meta.ncon > 0
      KKT
    elseif has_bounds(model)
      optim_check_bounded
    else
      unconstrained_check
    end,
    max_iter = meta.subsolver_max_iter,
    atol = meta.atol_sub(atol), # max(0.1, atol),# atol / 100,
    rtol = meta.rtol_sub(rtol), # max(0.1, rtol), #rtol / 100,
    optimality0 = one(T),
    unbounded_threshold = meta.subpb_unbounded_threshold,
  )
  return FPSSSolver(
    stp,
    meta,
    workspace,
    qdsolver,
    subproblem_solver,
    sub_stats,
    feasibility_solver,
    model,
    sub_stp,
  )
end

function SolverCore.reset!(solver::FPSSSolver)
  reinit!(solver.stp)
  reinit!(solver.sub_stp)
  NLPModels.reset!(solver.model)
  solver
end

function SolverCore.reset!(solver::FPSSSolver, nlp::AbstractNLPModel)
  @assert nlp.meta.nvar == solver.stp.pb.meta.nvar
  @assert nlp.meta.ncon == solver.stp.pb.meta.ncon
  SolverCore.reset!(solver)
  solver.stp.pb = nlp
  reinit!(solver.stp)
  solver.model.nlp = nlp
  solver
end
