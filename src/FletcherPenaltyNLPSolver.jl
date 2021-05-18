module FletcherPenaltyNLPSolver

using FastClosures, LinearAlgebra, Logging, SparseArrays

using Memoize

# JSO packages
using Krylov, LinearOperators, LDLFactorizations, NLPModels, NLPModelsModifiers, SolverCore
using Stopping

using NLPModelsIpopt

const is_knitro_installed = try
  @eval using NLPModelsKnitro
  true
catch
  #@warn "KNITRO not properly installed. Please run `] build KNITRO`"
  false
end

##################################################################
# Temporary
include("StoppingInterfacemod.jl")
#using StoppingInterface #ipopt, knitro, status_stopping_to_stats
#
###################################################################

include("model-Fletcherpenaltynlp.jl")

export FletcherPenaltyNLP
export obj, objgrad, objgrad!, grad!, grad
export hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!

function Fletcher_penalty_optimality_check(pb::AbstractNLPModel, state::NLPAtX)
  #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
  #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
  # returns i) + ii) OR iii) ?
  nxk = max(norm(state.x), 1.0)
  nlk = isnothing(state.lambda) ? 1.0 : max(norm(state.lambda), 1.0)

  cx = state.cx / nxk
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

###############################
#
# TO BE REMOVED
#
include("lbfgs.jl") # relies on SolverTools
#
#
###############################

include("feasibility.jl")

export fps_solve

"""
Solver for equality constrained non-linear programs based on Fletcher's penalty function.

    Cite: Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
    Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
    SIAM Journal on Scientific Computing, 42(3), A1809-A1835.

`fps_solve(:: NLPStopping, :: AbstractVector{T};  σ_0 :: Number = one(T), σ_max :: Number = 1/eps(T), σ_update :: Number = T(1.15), unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`
or
`fps_solve(:: AbstractNLPModel, :: AbstractVector{T}, σ_0 :: Number = one(T), σ_max :: Number = 1/eps(T), σ_update :: Number = T(1.15), unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`

Notes:     
- If the problem has inequalities, we use slack variables to get only equalities and bounds.
- `stp.current_state.res` contains the gradient of Fletcher's penalty function.
- `unconstrained_solver` must take an NLPStopping as input.

TODO:
- une façon robuste de mettre à jour le paramètre de pénalité. [Convergence to infeasible stationary points]
- Extend to bounds and inequality constraints.
- Handle the tol_check from the paper !
- Continue to explore the paper.
- [Long term] Complemetarity constraints
"""
function fps_solve(nlp::AbstractNLPModel, x0::AbstractVector{T} = nlp.meta.x0; kwargs...) where {T}
  if !(nlp.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  if has_inequalities(nlp)
    nlp = SlackModel(nlp)
  end
  #meta = AlgoData(T; kwargs...)
  meta = FPSSSolver(nlp, T(0); kwargs...)

  cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
  #Tanj: how to handle stopping criteria where tol_check depends on the State?
  Fptc(atol, rtol, opt0) =
    rtol * vcat(ones(nlp.meta.ncon) .+ norm(cx0, Inf), ones(nlp.meta.nvar) .+ norm(gx0, Inf))

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
    atol = T(1e-6), # really convert here ?
    rtol = T(1e-6),
    tol_check = Fptc,
    max_cntrs = Stopping._init_max_counters(allevals = typemax(Int64));
    kwargs...,
  )

  return fps_solve(stp, meta)
end

function fps_solve(stp::NLPStopping; kwargs...)
  T = eltype(stp.pb.meta.x0)
  meta = AlgoData(T; kwargs...)

  return fps_solve(stp, meta)
end

include("algo.jl")

end #end of module
