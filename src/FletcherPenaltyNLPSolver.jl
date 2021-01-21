module FletcherPenaltyNLPSolver

using LinearAlgebra, Logging, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

@warn "Depend on the last versions of Stopping.jl (≥ 0.2.5) - https://github.com/vepiteski/Stopping.jl"

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
                 (:TimeLimit, :max_time), #(:Tired, :max_time),
                 (:EvaluationLimit, :max_eval), #(:ResourcesExhausted, :max_eval),
                 (:ResourcesOfMainProblemExhausted, :max_eval),
                 (:Infeasible, :infeasible),
                 (:DomainError, :exception),
                 (:StopByUser, :unknown),
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
"""
MOVE THIS FUNCTION TO STOPPING
knitro(nlp) DOESN'T CHECK THE WRONG KWARGS, AND RETURN AN ERROR.

knitro(::NLPStopping)

Selection of possible [options](https://www.artelys.com/docs/knitro/3_referenceManual/userOptions.html):
*General options*
algorithm: Indicates which algorithm to use to solve the problem
blasoption: Specifies the BLAS/LAPACK function library to use for basic vector 
            and matrix computations
cg_maxit: Determines the maximum allowable number of inner conjugate gradient 
          (CG) iterations
cg_pmem: Specifies number of nonzero elements per hessian column when computing 
         preconditioner
cg_precond: Specifies whether or not to apply preconditioning during 
            CG iterations in barrier algorithms
cg_stoptol: Relative stopping tolerance for CG subproblems
convex: Identify convex models and apply specializations often beneficial for convex models
delta: Specifies the initial trust region radius scaling factor
eval_fcga: Specifies that gradients are provided together with functions in one callback
honorbnds: Indicates whether or not to enforce satisfaction of simple variable bounds
initpenalty: Initial penalty value used in Knitro merit function
linesearch_maxtrials: Indicates the maximum allowable number of trial points during the linesearch
linesearch: Indicates which linesearch strategy to use for the Interior/Direct or SQP algorithm
linsolver_ooc: Indicates whether to use Intel MKL PARDISO out-of-core solve of linear systems
linsolver: Indicates which linear solver to use to solve linear systems arising in Knitro algorithms
linsolver_maxitref
linsolver_pivottol
objrange: Specifies the extreme limits of the objective function for purposes of determining unboundedness
          Default value: 1.0e20
presolve: Determine whether or not to use the Knitro presolver
presolve_initpt: Controls whether Knitro presolver can shift user-supplied initial point
restarts: Specifies whether to enable automatic restarts
restarts_maxit: Maximum number of iterations before restarting when restarts are enabled
scale: Specifies whether to perform problem scaling
soc: Specifies whether or not to try second order corrections (SOC)
strat_warm_start: Specifies whether or not to invoke a warm-start strategy

*Derivatives options*
bfgs_scaling: Specifies the initial scaling for the BFGS or L-BFGS Hessian approximation
derivcheck: Determine whether or not to perform a derivative check on the model
gradopt: Specifies how to compute the gradients of the objective and constraint functions
hessian_no_f: Determines whether or not to allow Knitro to request Hessian 
              evaluations without the objective component included.
hessopt: Specifies how to compute the (approximate) Hessian of the Lagrangian
         1 (exact) User provides a routine for computing the exact Hessian. (Default)
         4 (product_findiff) Knitro computes Hessian-vector products using finite-differences.
         5 (product) User provides a routine to compute the Hessian-vector products.
         6 (lbfgs) Knitro computes a limited-memory quasi-Newton BFGS Hessian 
                   (its size is determined by the option lmsize).
lmsize: Specifies the number of limited memory pairs stored when approximating the Hessian

*Termination options*
feastol: Specifies the final relative stopping tolerance for the feasibility error.
         Default value: 1.0e-6
feastol_abs: Specifies the final absolute stopping tolerance for the feasibility error.
             Default value: 1.0e-3
fstopval: Used to implement a custom stopping condition based on the objective function value
ftol: The optimization process will terminate if feasible and the relative change 
      in the objective function is less than ftol
      Default value: 1.0e-15
ftol_iters: The optimization process will terminate if the relative change in 
            the objective function is less than ftol for ftol_iters consecutive 
            feasible iterations. Default value: 5

maxfevals: Specifies the maximum number of function evaluations before termination.
           Default value: -1 (unlimited)
maxit: Specifies the maximum number of iterations before termination
       0 is default value, let Knitro set it (10000 for LP/NLP)
maxtime_cpu: Specifies, in seconds, the maximum allowable CPU time before termination. 
             Default value: 1.0e8
maxtime_real: Specifies, in seconds, the maximum allowable real time before termination. 
              Default value: 1.0e8
opttol: Specifies the final relative stopping tolerance for the KKT (optimality) error
        Default value: 1.0e-6
opttol_abs: Specifies the final absolute stopping tolerance for the KKT (optimality) error
            Default value: 1.0e-3
xtol: The optimization process will terminate if the relative change of the 
      solution point estimate is less than xtol. Default value: 1.0e-12
xtol_iters: Number of consecutive iterations where change of the solution point 
            estimate is less than xtol before Knitro stops.
            Default is 1
            
*Output options*
out_hints: Print diagnostic hints (e.g. on user option settings) after solving
           0 no prints, and 1 prints (default in Knitro)
outlev: Controls the level of output produced by Knitro
        0 (none) Printing of all output is suppressed.
        1 (summary) Print only summary information.
        2 (iter_10) Print basic information every 10 iterations. (default in Knitro)
"""
function knitro(stp          :: NLPStopping;
                convex       :: Int  = -1, #let Knitro deal with it :)
                objrange     :: Real = stp.meta.unbounded_threshold,
                hessopt      :: Int  = 1,
                feastol      :: Real = stp.meta.rtol,
                feastol_abs  :: Real = stp.meta.atol,
                opttol       :: Real = stp.meta.rtol,
                opttol_abs   :: Real = stp.meta.atol,
                maxfevals    :: Int  = stp.meta.max_cntrs[:neval_sum],
                maxit        :: Int  = 0, #stp.meta.max_iter
                maxtime_real :: Real = stp.meta.max_time,
                out_hints    :: Int  = 0,
                outlev       :: Int  = 0,
                kwargs...)
    
    @assert -1 ≤ convex ≤ 1
    @assert 1  ≤ hessopt ≤ 7            
    @assert 0  ≤ out_hints ≤ 1
    @assert 0  ≤ outlev ≤ 6
    @assert 0  ≤ maxit
    
    nlp = stp.pb #FletcherPenaltyNLP
    #y0 = stp.current_state.lambda #si défini
    #z0 = stp.current_state.mu #si défini 
    stats = knitro(nlp, x0           = stp.current_state.x,
                        objrange     = objrange,
                        feastol      = feastol,
                        feastol_abs  = feastol_abs,
                        opttol       = opttol,
                        opttol_abs   = opttol_abs,
                        maxfevals    = maxfevals,
                        maxit        = maxit,
                        maxtime_real = maxtime_real,
                        out_hints    = out_hints,
                        outlev       = outlev;
                        kwargs...)
    
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
