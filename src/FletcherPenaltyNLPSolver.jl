module FletcherPenaltyNLPSolver

using LinearAlgebra, Logging, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

@warn "Depend on the last versions of Stopping.jl (≥ 0.2.5) - https://github.com/vepiteski/Stopping.jl"

using Stopping #> 0.2.1
using StoppingInterface #ipopt, knitro, status_stopping_to_stats

include("model-Fletcherpenaltynlp.jl")

export FletcherPenaltyNLP
export obj, objgrad, objgrad!, grad!, grad
export hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!

function Fletcher_penalty_optimality_check(pb :: AbstractNLPModel, state :: NLPAtX)
    #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
    #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
    #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
    # returns i) + ii) OR iii) ?
    nxk = max(norm(state.x), 1.)
    nlk = isnothing(state.lambda) ? 1. : max(norm(state.lambda), 1.)
    
    cx  = state.cx/nxk
    res = state.res/nlk
    
 return vcat(cx, res)
end

include("parameters.jl")

###############################
#
# TO BE REMOVED
#
include("lbfgs.jl")
#
#
###############################

export Fletcher_penalty_solver

"""
Solver for equality constrained non-linear programs based on Fletcher's penalty function.

    Cite: Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
    Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
    SIAM Journal on Scientific Computing, 42(3), A1809-A1835.

`Fletcher_penalty_solver(:: NLPStopping, :: AbstractVector{T};  σ_0 :: Number = one(T), σ_max :: Number = 1/eps(T), σ_update :: Number = T(1.15), linear_system_solver :: Function  = _solve_with_linear_operator, unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`
or
`Fletcher_penalty_solver(:: AbstractNLPModel, :: AbstractVector{T}, σ_0 :: Number = one(T), σ_max :: Number = 1/eps(T), σ_update :: Number = T(1.15), linear_system_solver :: Function = _solve_with_linear_operator, unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`

Notes:
- stp.current_state.res contains the gradient of Fletcher's penalty function.
- unconstrained\\_solver must take an NLPStopping as input.
- *linear\\_system\\_solver* solves two linear systems with different rhs following the format:
*linear\\_system\\_solver(nlp, x, rhs1, rhs2; kwargs...)*
List of implemented methods:
i)   \\_solve\\_system\\_dense
ii)  \\_solve\\_with\\_linear\\_operator
iii) \\_solve\\_system\\_factorization\\_eigenvalue
iv)  \\_solve\\_system\\_factorization\\_lu

TODO:
- une façon robuste de mettre à jour le paramètre de pénalité. [Convergence to infeasible stationary points]
- Extend to bounds and inequality constraints.
- Handle the tol_check from the paper !
- Use Hessian (approximation) from FletcherPenaltyNLP
- Continue to explore the paper.
- [Long term] Complemetarity constraints
"""
function Fletcher_penalty_solver(nlp                   :: AbstractNLPModel;
                                 x0                    :: AbstractVector = nlp.meta.x0,
                                 rtol                  :: Number    = 1e-6,
                                 σ_0                   :: Number    = 1.,
                                 σ_max                 :: Number    = 1/eps(),
                                 σ_update              :: Number    = 1.15,
                                 ρ_0                   :: Number    = 1.,
                                 ρ_max                 :: Number    = 1/eps(),
                                 ρ_update              :: Number    = 1.15,
                                 δ_0                   :: Number    = √eps(),
                                 linear_system_solver  :: Function  = _solve_with_linear_operator,
                                 unconstrained_solver  :: Function  = knitro,
                                 hessian_approx        :: Int       = 2,
                                 kwargs...)

 cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
 #Tanj: how to handle stopping criteria where tol_check depends on the State?
 Fptc(atol, rtol, opt0) = rtol * vcat(ones(nlp.meta.ncon) .+ norm(cx0, Inf),
                                      ones(nlp.meta.nvar) .+ norm(gx0, Inf))
                                      
 initial_state = NLPAtX(x0, 
                        zeros(nlp.meta.ncon), 
                        Array{Float64,1}(undef, nlp.meta.ncon+nlp.meta.nvar), 
                        cx = cx0, 
                        gx = gx0, 
                        res = gx0)
 stp = NLPStopping(nlp, initial_state,
                   optimality_check = Fletcher_penalty_optimality_check,
                   rtol = rtol,
                   tol_check = Fptc,
                   max_cntrs = Stopping._init_max_counters(allevals = typemax(Int64)); kwargs...)

 return Fletcher_penalty_solver(stp,
                                σ_0 = σ_0, σ_max = σ_max, σ_update = σ_update,
                                ρ_0 = ρ_0, ρ_max = ρ_max, ρ_update = ρ_update,
                                δ_0 = δ_0,
                                linear_system_solver = linear_system_solver,
                                hessian_approx       = hessian_approx,
                                unconstrained_solver = unconstrained_solver)
end

include("algo.jl")

end #end of module
