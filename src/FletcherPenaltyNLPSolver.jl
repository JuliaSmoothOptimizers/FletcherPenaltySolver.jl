module FletcherPenaltyNLPSolver

using LinearAlgebra, Logging, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

@warn "Depend on the last versions of Stopping.jl (≥ 0.2.4) - https://github.com/vepiteski/Stopping.jl"

using Stopping #> 0.2.1

include("model-Fletcherpenaltynlp.jl")

export FletcherPenaltyNLP
export obj, objgrad, objgrad!, grad!, grad
export hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!

"""
Move this function to Stopping.jl
"""
function status_stopping_to_stats(stp :: AbstractStopping)
 stp_status = status(stp)
 convert = Dict([(:Optimal, :first_order),
                 (:SubProblemFailure, :unknown),
                 (:SubOptimal, :acceptable),
                 (:Unbounded, :unbounded),
                 (:UnboundedPb, :unbounded),
                 (:Stalled, :stalled),
                 (:IterationLimit, :max_iter),
                 (:Tired, :max_time),
                 (:ResourcesExhausted, :max_eval),
                 (:ResourcesOfMainProblemExhausted, :max_eval),
                 (:Infeasible, :infeasible),
                 (:DomainError, :exception),
                 (:Unknown, :unknown)
                 ])
 return convert[stp_status]
end

"""
copy-paste from the coming Stopping-branch
https://github.com/vepiteski/Stopping.jl/blob/type-stable-version/src/Stopping/NLPStoppingmod.jl
"""
function _init_max_counters(; quick  :: T = 20000,
                              obj    :: T = quick,
                              grad   :: T = quick,
                              cons   :: T = quick,
                              jcon   :: T = quick,
                              jgrad  :: T = quick,
                              jac    :: T = quick,
                              jprod  :: T = quick,
                              jtprod :: T = quick,
                              hess   :: T = quick,
                              hprod  :: T = quick,
                              jhprod :: T = quick,
                              sum    :: T = quick*11) where {T <: Int}

  cntrs = Dict{Symbol,T}([(:neval_obj,       obj), (:neval_grad,   grad),
                          (:neval_cons,     cons), (:neval_jcon,   jcon),
                          (:neval_jgrad,   jgrad), (:neval_jac,    jac),
                          (:neval_jprod,   jprod), (:neval_jtprod, jtprod),
                          (:neval_hess,     hess), (:neval_hprod,  hprod),
                          (:neval_jhprod, jhprod), (:neval_sum,    sum)])

 return cntrs
end

include("algo.jl")

import NLPModelsKnitro: knitro
#Check https://github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl/blob/master/src/NLPModelsKnitro.jl
function knitro(stp :: NLPStopping)
    
    nlp = stp.pb #FletcherPenaltyNLP
    stats = knitro(nlp, x0 = stp.current_state.x)
    
    if stats.status ∈ (:first_order, :acceptable) 
       stp.meta.optimal = true
       
       stp.current_state.x  = stats.solution
       stp.current_state.fx = stats.objective
       stp.current_state.gx = grad(nlp, stats.solution)#stats.dual_feas
       stp.current_state.current_score  = norm(stp.current_state.gx, Inf)#stats.dual_feas
    elseif stats.status == :stalled
        stp.meta.stalled = true #point is feasible
    elseif stats.status == :infeasible
        stp.meta.infeasible = true #euhhhh, wait ... isn't it unconstrained?
    elseif stats.status == :unbounded
        #grrrrr
        stp.meta.unbounded = true
    elseif stats.status == :max_iter
        stp.meta.iteration_limit = true
    elseif stats.status == :max_time
        stp.meta.tired = true
    elseif stats.status == :max_eval  
        stp.meta.resources = true
    else #stats.status ∈ (:exception, :unknown)  
        #Ouch...
        @show stats.status
        stp.meta.fail_sub_pb
    end
    
    return stp
end

import NLPModelsIpopt: ipopt
#Using Stopping, the idea is to create a buffer function
function ipopt(stp :: NLPStopping) #kwargs

 #xk = solveIpopt(stop.pb, stop.current_state.x)
 nlp = stp.pb
 stats = ipopt(nlp, print_level     = 0,
                    tol             = stp.meta.rtol,
                    x0              = stp.current_state.x,
                    max_iter        = stp.meta.max_iter,
                    max_cpu_time    = stp.meta.max_time,
                    dual_inf_tol    = stp.meta.atol,
                    constr_viol_tol = stp.meta.atol,
                    compl_inf_tol   = stp.meta.atol)

 #Update the meta boolean with the output message
 if stats.status == :first_order stp.meta.suboptimal      = true end
 if stats.status == :acceptable  stp.meta.suboptimal      = true end
 if stats.status == :infeasible  stp.meta.infeasible      = true end
 if stats.status == :small_step  stp.meta.stalled         = true end
 if stats.status == :max_iter    stp.meta.iteration_limit = true end
 if stats.status == :max_time    stp.meta.tired           = true end

 stp.meta.nb_of_stop = stats.iter
 #stats.elapsed_time

 x = stats.solution

 #Not mandatory, but in case some entries of the State are used to stop
 fill_in!(stp, x) #too slow

 stop!(stp)

 return stp
end

end #end of module
