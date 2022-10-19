module FletcherPenaltySolver

using FastClosures, LinearAlgebra, Logging, SparseArrays

# JSO packages
using Krylov, LinearOperators, LDLFactorizations, NLPModels, NLPModelsModifiers, SolverCore
using JSOSolvers, NLPModelsIpopt, NLPModelsKnitro
using Stopping, StoppingInterface

include("model-Fletcherpenaltynlp.jl")

export FletcherPenaltyNLP
export obj, objgrad, objgrad!, grad!, grad
export hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!

"""
    Fletcher_penalty_optimality_check(pb::AbstractNLPModel, state::NLPAtX)

Optimality function used by default in the algorithm.
An alternative is to use the function `KKT` from `Stopping.jl`.

The function returns a vector of length ncon + nvar containing:
  * |c(x) - lcon| / |x|₂
  * res / |λ|₂ ; x - max(min(x - res, uvar), lvar)) if it has bounds

The fields `x`, `cx` and `res` need to be filled. If `state.lambda` is `nothing` then we take |λ|₂=1.
"""
function Fletcher_penalty_optimality_check(pb::AbstractNLPModel{T, S}, state::NLPAtX) where {T, S}
  cx = view(state.current_score, 1:get_ncon(pb))
  res = view(state.current_score, (1 + get_ncon(pb)):(get_ncon(pb) + get_nvar(pb)))
  #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
  #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  # returns i) + ii) OR iii) ?
  nxk = max(norm(state.x), one(T))
  nlk = isnothing(state.lambda) ? one(T) : max(norm(state.lambda), one(T))

  cx .= max.(state.cx .- get_ucon(pb), get_lcon(pb) .- state.cx, 0) ./ nxk
  if has_bounds(pb) # && state.mu == []
    res .= state.x .- max.(min.(state.x .- state.res, pb.meta.uvar), pb.meta.lvar)
    # elseif has_bounds(pb)
    #  gix = min.(state.x - pb.meta.lvar, pb.meta.uvar - state.x)
    #  mu = state.mu
    #  res = max.(state.res, min.(mu, gix, mu .* gix))
  else
    res .= state.res ./ nlk
  end

  return state.current_score
end

include("parameters.jl")
export AlgoData, FPSSSolver

include("feasibility.jl")

export fps_solve, solve!

"""
    fps_solve(nlp::AbstractNLPModel, x0::AbstractVector{T} = nlp.meta.x0; subsolver_verbose::Int = 0, kwargs...)

Compute a local minimum of a bound and equality-constrained optimization problem using Fletcher's penalty function and the implementation described in

    Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
    Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
    SIAM Journal on Scientific Computing, 42(3), A1809-A1835.
    https://doi.org/10.1137/19M1238265

For advanced usage, the principal call to the solver uses a `NLPStopping`, see `Stopping.jl`.

    fps_solve(stp::NLPStopping, fpssolver::FPSSSolver{T, QDS, US}; subsolver_verbose::Int = 0)
    fps_solve(stp::NLPStopping; subsolver_verbose::Int = 0, kwargs...)

# Arguments
- `nlp::AbstractNLPModel`: the model solved, see `NLPModels.jl`;
- `x`: Initial guess. If `x` is not specified, then `nlp.meta.x0` is used.

# Keyword arguments
- `fpssolver`: see [`FPSSSolver`](@ref);
- `verbose::Int = 0`: if > 0, display iteration information of the solver;
- `subsolver_verbose::Int = 0`: if > 0, display iteration information of the subsolver;

All the information regarding stopping criteria can be set in the `NLPStopping` object.
Additional `kwargs` are given to the `NLPStopping`.
By default, the optimality condition used to declare optimality is [`Fletcher_penalty_optimality_check`](@ref).

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

If one define a `Stopping` before calling `fps_solve`, it is possible to access all the information computed by the algorithm.

# Notes

- If the problem has inequalities, we use slack variables to get only equalities and bounds via `NLPModelsModifiers.jl`.
- `stp.current_state.res` contains the gradient of Fletcher's penalty function.
- `subproblem_solver` must take an `NLPStopping` as input, see `StoppingInterface.jl`.

# Examples
```julia
julia> using FletcherPenaltySolver, ADNLPModels
julia> nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0]);
julia> stats = fps_solve(nlp)
"Execution stats: first-order stationary"
```
"""
function fps_solve(
  nlp::AbstractNLPModel{T, V},
  x0::V = nlp.meta.x0;
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  kwargs...,
) where {T, V}
  if !(nlp.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  ineq = has_inequalities(nlp)
  ns = nlp.meta.ncon - length(nlp.meta.jfix)
  if ineq
    x0 = vcat(x0, fill!(V(undef, ns), zero(T)))
    nlp = SlackModel(nlp)
  end

  meta = FPSSSolver(nlp, x0; kwargs...)
  stats =
    SolverCore.solve!(meta, meta.stp; verbose = verbose, subsolver_verbose = subsolver_verbose)
  if ineq && stats.multipliers_L != []
    nnvar = nlp.model.meta.nvar
    # reshape the stats to fit the original problem
    stats = GenericExecutionStats(
      nlp.model,
      status = stats.status,
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
      nlp.model,
      status = stats.status,
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

function fps_solve(stp::NLPStopping; verbose::Int = 0, subsolver_verbose::Int = 0, kwargs...)
  meta = FPSSSolver(stp; kwargs...)
  # Update the state
  x = stp.current_state.x
  fill_in!(stp, x, Hx = stp.current_state.Hx)

  SolverCore.solve!(meta, stp; verbose = verbose, subsolver_verbose = subsolver_verbose)
end

include("algo.jl")

end #end of module
