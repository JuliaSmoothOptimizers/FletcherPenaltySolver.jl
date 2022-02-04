#module StoppingInterface

using LinearAlgebra, Stopping

"""
Return the status in GenericStats from a Stopping.
"""
function status_stopping_to_stats(stp::AbstractStopping)
  stp_status = status(stp)
  convert = Dict([
    (:Optimal, :first_order),
    (:SubProblemFailure, :unknown),
    (:SubOptimal, :acceptable),
    (:Unbounded, :unbounded),
    (:UnboundedPb, :unbounded),
    (:Stalled, :stalled),
    (:IterationLimit, :max_iter),
    (:Tired, :max_time), #disapear from Stopping.jl#v0.2.5
    (:TimeLimit, :max_time),
    (:ResourcesExhausted, :max_eval), #disapear from Stopping.jl#v0.2.5
    (:EvaluationLimit, :max_eval),
    (:ResourcesOfMainProblemExhausted, :max_eval),
    (:Infeasible, :infeasible),
    (:DomainError, :exception),
    (:StopByUser, :user),
    (:Exception, :exception),
    (:Unknown, :unknown),
  ])
  return convert[stp_status]
end

"""
Initialize a GenericStats from Stopping
"""
function stopping_to_stats(stp::NLPStopping)
  nlp = stp.pb
  cx = stp.current_state.cx
  return GenericExecutionStats(
    status_stopping_to_stats(stp),
    stp.pb,
    solution = stp.current_state.x,
    objective = stp.current_state.fx,
    primal_feas = max.(cx - get_ucon(nlp), get_lcon(nlp) - cx, 0),
    dual_feas = norm(stp.current_state.res, Inf),
    multipliers = stp.current_state.lambda,
    iter = stp.meta.nb_of_stop,
    elapsed_time = stp.current_state.current_time - stp.meta.start_time,
  )
end

function stats_status_to_meta!(stp::AbstractStopping, stats::GenericExecutionStats)
  return stats_status_to_meta!(stp, stats.status)
end

function stats_status_to_meta!(stp::AbstractStopping, status::Symbol)
  #Update the meta boolean with the output message
  if status == :first_order
    stp.meta.optimal = true
  end
  if status == :acceptable
    stp.meta.suboptimal = true
  end
  if status == :infeasible
    stp.meta.infeasible = true
  end
  if status == :small_step
    stp.meta.stalled = true
  end
  if status == :max_eval
    stp.meta.resources = true
  end
  if status == :max_iter
    stp.meta.iteration_limit = true
  end
  if status == :max_time
    stp.meta.tired = true
  end
  if status ∈ [:neg_pred, :not_desc]
    stp.meta.fail_sub_pb = true
  end
  if status == :unbounded
    stp.meta.unbounded = true
  end
  if status == :user
    stp.meta.stopbyuser = true
  end
  if status ∈ [:stalled, :small_residual, :small_step]
    stp.meta.stalled = true
  end
  if status == :exception
    stp.meta.exception = true
  end #available ≥ 0.2.6

  return stp
end
export status_stopping_to_stats, stopping_to_stats

#using NLPModelsIpopt

"""
ipopt(nlp) DOESN'T CHECK THE WRONG KWARGS, AND RETURN AN ERROR.
ipopt(::NLPStopping)
"""
function NLPModelsIpopt.ipopt(stp::NLPStopping; subsolver_verbose::Int = 0, kwargs...)

  #xk = solveIpopt(stop.pb, stop.current_state.x)
  nlp = stp.pb
  stats = ipopt(
    nlp,
    print_level = subsolver_verbose,
    tol = stp.meta.rtol,
    x0 = stp.current_state.x,
    max_iter = stp.meta.max_iter,
    max_cpu_time = stp.meta.max_time,
    dual_inf_tol = stp.meta.atol,
    constr_viol_tol = stp.meta.atol,
    compl_inf_tol = stp.meta.atol,
    kwargs...,
  )

  #Update the meta boolean with the output message
  #=
  if stats.status == :first_order stp.meta.suboptimal      = true end
  if stats.status == :acceptable  stp.meta.suboptimal      = true end
  if stats.status == :infeasible  stp.meta.infeasible      = true end
  if stats.status == :small_step  stp.meta.stalled         = true end
  if stats.status == :max_eval    stp.meta.max_eval        = true end
  if stats.status == :max_iter    stp.meta.iteration_limit = true end
  if stats.status == :max_time    stp.meta.tired           = true end
  if stats.status ∈ [:neg_pred, 
                     :not_desc]   stp.meta.fail_sub_pb     = true end
  if stats.status == :unbounded   stp.meta.unbounded       = true end
  if stats.status == :user        stp.meta.stopbyuser      = true end
  if stats.status ∈ [:stalled, 
                     :small_residual,
                     :small_step]   stp.meta.stalled       = true end
  #if stats.status == :exception   stp.meta.exception       = true end #available ≥ 0.2.6
  =#
  stp = stats_status_to_meta!(stp, stats)

  if status(stp) == :Unknown
    @warn "Error in StoppingInterface statuses: return status is $(stats.status)"
    @show stats.solver_specific
  end

  stp.meta.nb_of_stop = stats.iter
  #stats.elapsed_time

  x = stats.solution

  #Not mandatory, but in case some entries of the State are used to stop
  fill_in!(stp, x) #too slow

  stop!(stp)

  return stp
end

@init begin
  @require KNITRO = "67920dd8-b58e-52a8-8622-53c4cffbe346" begin
    @require NLPModelsKnitro = "bec4dd0d-7755-52d5-9a02-22f0ffc7efcb" begin
      is_knitro_installed = true

      """
      knitro(nlp) DOESN'T CHECK THE WRONG KWARGS, AND RETURN AN ERROR.
      knitro(::NLPStopping)
      Selection of possible [options](https://www.artelys.com/docs/knitro/3_referenceManual/userOptions.html):
      """
      function NLPModelsKnitro.knitro(
        stp::NLPStopping;
        convex::Int = -1, #let Knitro deal with it :)
        objrange::Real = stp.meta.unbounded_threshold,
        hessopt::Int = 1,
        feastol::Real = stp.meta.rtol,
        feastol_abs::Real = stp.meta.atol,
        opttol::Real = stp.meta.rtol,
        opttol_abs::Real = stp.meta.atol,
        maxfevals::Int = min(stp.meta.max_cntrs[:neval_sum], typemax(Int32)),
        maxit::Int = 0, #stp.meta.max_iter
        maxtime_real::Real = stp.meta.max_time,
        out_hints::Int = 0,
        subsolver_verbose::Int = 0, #1 to see everything
        algorithm::Int = 0, # *New* 2
        ftol::Real = 1.0e-15, # *New*
        ftol_iters::Int = 5, # *New*
        xtol::Real = stp.meta.atol, # 1.0e-12, # *New*
        xtol_iters::Int = 2, # 3 # *New*
        kwargs...,
      )
        @assert -1 ≤ convex ≤ 1
        @assert 1 ≤ hessopt ≤ 7
        @assert 0 ≤ out_hints ≤ 1
        @assert 0 ≤ subsolver_verbose ≤ 6
        @assert 0 ≤ maxit

        nlp = stp.pb
        #y0 = stp.current_state.lambda #si défini
        #z0 = stp.current_state.mu #si défini 
        solver = NLPModelsKnitro.KnitroSolver(
          nlp,
          x0 = stp.current_state.x,
          objrange = objrange,
          feastol = feastol,
          feastol_abs = feastol_abs,
          opttol = opttol,
          opttol_abs = opttol_abs,
          maxfevals = maxfevals,
          maxit = maxit,
          maxtime_real = maxtime_real,
          out_hints = out_hints,
          algorithm = algorithm,
          ftol = ftol,
          ftol_iters = ftol_iters,
          xtol = xtol,
          xtol_iters = xtol_iters,
          outlev = subsolver_verbose;
          kwargs...,
        )
        stats = NLPModelsKnitro.knitro!(nlp, solver)
        #@show stats.status, stats.solver_specific[:internal_msg]
        #if stats.status ∉ (:unbounded, :exception, :unknown) #∈ (:first_order, :acceptable) 
        stp.current_state.x = stats.solution
        stp.current_state.fx = stats.objective
        stp.current_state.gx = KNITRO.KN_get_objgrad_values(solver.kc)[2]
        #norm(stp.current_state.gx, Inf)#stats.dual_feas #TODO: this is for unconstrained problem!!
        stp.current_state.mu = stats.multipliers_L
        stp.current_state.current_score = max(stats.dual_feas, stats.primal_feas)
        #end
        #Update the meta boolean with the output message
        stp = stats_status_to_meta!(stp, stats)
        #@show status(stp, list = true)
        if status(stp) == :Unknown
          @warn "Error in StoppingInterface statuses: return status is $(stats.status)"
          #print(stats)
        end

        return stp #would be better to return the stats somewhere
      end
    end
  end
end

#end # module
