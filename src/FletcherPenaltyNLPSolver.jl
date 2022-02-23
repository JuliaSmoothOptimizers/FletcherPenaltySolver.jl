module FletcherPenaltyNLPSolver

using FastClosures, LinearAlgebra, Logging, SparseArrays

# JSO packages
using Krylov, LinearOperators, LDLFactorizations, NLPModels, NLPModelsModifiers, SolverCore
using Stopping, StoppingInterface

include("model-Fletcherpenaltynlp.jl")

export FletcherPenaltyNLP
export obj, objgrad, objgrad!, grad!, grad
export hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!

"""
    Fletcher_penalty_optimality_check(pb::AbstractNLPModel, state::NLPAtX)

Optimality function used by default in the algorithm.
An alternative is to use the function `KKT` from `Stopping.jl`.
"""
function Fletcher_penalty_optimality_check(pb::AbstractNLPModel{T, S}, state::NLPAtX) where {T, S}
  #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
  #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  # returns i) + ii) OR iii) ?
  nxk = max(norm(state.x), one(T))
  nlk = isnothing(state.lambda) ? one(T) : max(norm(state.lambda), one(T))

  cx = abs.(state.cx - get_lcon(pb)) / nxk # max.(cx - get_ucon(nlp), get_lcon(nlp) - cx, 0) / nxk
  if has_bounds(pb) # && state.mu == []
    proj = max.(min.(state.x - state.res, pb.meta.uvar), pb.meta.lvar)
    res = state.x - proj
    # elseif has_bounds(pb)
    #  gix = min.(state.x - pb.meta.lvar, pb.meta.uvar - state.x)
    #  mu = state.mu
    #  res = max.(state.res, min.(mu, gix, mu .* gix))
  else
    res = state.res / nlk
  end

  return vcat(cx, res)
end

include("parameters.jl")
export AlgoData, FPSSSolver

include("feasibility.jl")

export fps_solve

"""
    fps_solve(nlp::AbstractNLPModel, x0::AbstractVector{T} = nlp.meta.x0; subsolver_verbose::Int = 0, lagrange_bound = 1 / sqrt(eps(T)), kwargs...)

Compute a local minimum of a bound and equality-constrained optimization problem using Fletcher's penalty function and the implementation described in

    Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
    Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
    SIAM Journal on Scientific Computing, 42(3), A1809-A1835.
    https://doi.org/10.1137/19M1238265

For advanced usage, the principal call to the solver uses a `NLPStopping`, see `Stopping.jl`.

    fps_solve(stp::NLPStopping, fpssolver::FPSSSolver{T, QDS, US}; subsolver_verbose::Int = 0, lagrange_bound::T = 1 / sqrt(eps(T)))
    fps_solve(stp::NLPStopping; subsolver_verbose::Int = 0, lagrange_bound = 1 / sqrt(eps()), kwargs...)

# Arguments
- `nlp::AbstractNLPModel`: the model solved, see `NLPModels.jl`;
- `x`: Initial guess. If `x` is not specified, then `nlp.meta.x0` is used.

# Keyword arguments
- `fpssolver`: see [`FPSSSolver`](@ref);
- `subsolver_verbose::Int = 0`: if > 0, display iteration information of the subsolver;
- `lagrange_bound = 1 / sqrt(eps())`: bound used to declare the Lagrange multiplier unbounded.

All the information regarding stopping criteria can be set in the `NLPStopping` object.
Additional `kwargs` are given to the `NLPStopping`.
By default, the optimality condition used to declare optimality is [`Fletcher_penalty_optimality_check`](@ref).

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

If one define a `Stopping` before calling `fps_solve`, it is possible to access all the information computed by the algorithm.

# Notes

- If the problem has inequalities, we use slack variables to get only equalities and bounds via `NLPModelsModifiers.jl`.
- `stp.current_state.res` contains the gradient of Fletcher's penalty function.
- `unconstrained_solver` must take an `NLPStopping` as input, see `StoppingInterface.jl`.

# Examples
```julia
julia> using FletcherPenaltyNLPSolver, ADNLPModels
julia> nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0]);
julia> stats = fps_solve(nlp)
"Execution stats: first-order stationary"
```
"""
function fps_solve(
  nlp::AbstractNLPModel,
  x0::AbstractVector{T} = nlp.meta.x0;
  subsolver_verbose::Int = 0,
  lagrange_bound = 1 / sqrt(eps(T)),
  kwargs...,
) where {T}
  if !(nlp.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  ineq = has_inequalities(nlp)
  ns = nlp.meta.ncon - length(nlp.meta.jfix)
  if ineq
    x0 = vcat(x0, zeros(T, ns))
    nlp = SlackModel(nlp)
  end
  meta = FPSSSolver(nlp, T(0); kwargs...)

  cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
  #Tanj: how to handle stopping criteria where tol_check depends on the State?
  Fptc(atol, rtol, opt0) =
    rtol * vcat(ones(T, nlp.meta.ncon) .+ norm(cx0, Inf), ones(T, nlp.meta.nvar) .+ norm(gx0, Inf))
  initial_state = NLPAtX(
    x0,
    zeros(T, nlp.meta.ncon),
    Array{T, 1}(undef, nlp.meta.ncon + nlp.meta.nvar),
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
  stats =
    fps_solve(stp, meta; subsolver_verbose = subsolver_verbose, lagrange_bound = lagrange_bound)
  if ineq && stats.multipliers_L != []
    nnvar = nlp.model.meta.nvar
    # reshape the stats to fit the original problem
    stats = GenericExecutionStats(
      stats.status,
      nlp.model,
      solution = stats.solution[1:nnvar],
      objective = stats.objective,
      primal_feas = stats.primal_feas,
      dual_feas = stats.dual_feas,
      multipliers = vcat(stats.multipliers, stats.multipliers_L[(nnvar + 1):(nnvar + ns)]),
      multipliers_L = stats.multipliers_L[1:nnvar],
      iter = stats.iter,
      elapsed_time = stats.elapsed_time,
      # solver_specific = stats.solver_specific,
    )
  elseif ineq
    nnvar = nlp.model.meta.nvar
    # reshape the stats to fit the original problem
    stats = GenericExecutionStats(
      stats.status,
      nlp.model,
      solution = stats.solution[1:nnvar],
      objective = stats.objective,
      primal_feas = stats.primal_feas,
      dual_feas = stats.dual_feas,
      multipliers = stats.multipliers,
      iter = stats.iter,
      elapsed_time = stats.elapsed_time,
      # solver_specific = stats.solver_specific,
    )
  end
  return stats
end

function fps_solve(
  stp::NLPStopping;
  subsolver_verbose::Int = 0,
  lagrange_bound = 1 / sqrt(eps()),
  kwargs...,
)
  nlp = stp.pb
  T = eltype(nlp.meta.x0)
  meta = FPSSSolver(nlp, T(0); kwargs...)
  # Update the state
  x = stp.current_state.x
  fill_in!(stp, x, Hx = stp.current_state.Hx)

  return fps_solve(
    stp,
    meta;
    subsolver_verbose = subsolver_verbose,
    lagrange_bound = T(lagrange_bound),
  )
end

include("algo.jl")

end #end of module
