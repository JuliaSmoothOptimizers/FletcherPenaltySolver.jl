function Fletcher_penalty_solver(stp  :: NLPStopping, 
                                 meta :: AlgoData{T}) where T

  state = stp.current_state
  #Initialize parameters
  x0, σ, ρ, δ = state.x, meta.σ_0, meta.ρ_0 , meta.δ_0
  
  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(stp.pb, σ, ρ, δ, meta.linear_system_solver, meta.hessian_approx)

  #First call to the stopping
  OK = start!(stp)
  #Prepare the subproblem-stopping for the unconstrained minimization.
  sub_stp = NLPStopping(nlp, NLPAtX(x0), main_stp = stp, 
                                         optimality_check = unconstrained_check,
                                         max_iter = 10000,
                                         atol     = 1e-4,
                                         rtol     = 1e-7)
  
  nc0 = norm(state.cx, Inf)
  Δ   = 0.95 #expected decrease in feasibility
  unsuccessful_subpb = 0 #number of consecutive failed subproblem solve.
  stalling = 0 #number of consecutive successful subproblem solve without progress
  restoration_phase = false
  
  @info log_header([:iter, :step, :f, :c, :score, :σ, :stat], [Int, String, T, T, T, T, Symbol],
                   hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :σ=>"σ"))
  @info log_row(Any[0, "Init", NaN, nc0, norm(state.current_score,Inf), σ, :Initial])

  while !OK #main loop

    #Solve the subproblem
    reinit!(sub_stp) #reinit the sub-stopping.
    sub_stp = meta.unconstrained_solver(sub_stp)
    #Update the State with the info given by the subproblem:
    if sub_stp.meta.optimal || sub_stp.meta.suboptimal
      if sub_stp.current_state.x == state.x
        stalling += 1  
      end
      unsuccessful_subpb = 0
      
      Stopping.update!(state, x      = sub_stp.current_state.x,
                              fx     = sub_stp.pb.fx,
                              gx     = sub_stp.pb.gx,
                              cx     = sub_stp.pb.cx,
                              lambda = sub_stp.pb.ys,
                              res    = grad(sub_stp.pb, sub_stp.current_state.x))
#@show "How good is this update ?", abs(state.fx - obj(stp.pb, state.x)), norm(state.gx - grad(stp.pb, state.x), norm(state.cx - cons(stp.pb, state.x)))
   elseif sub_stp.meta.unbounded
      unsuccessful_subpb += 1
   elseif sub_stp.meta.iteration_limit || sub_stp.meta.tired || sub_stp.meta.resources || sub_stp.meta.stalled
      unsuccessful_subpb += 1
   else #exception of unexpected failure
      stp.meta.fail_sub_pb = true
      @warn "Exception of unexpected failure: $(status(sub_stp, list = true))"
      #break 
   end
   
    #Check optimality conditions: either stop! is true OR the penalty parameter is too small
    stp.meta.fail_sub_pb = σ > meta.σ_max || ρ > meta.ρ_max #stp.meta.stalled 
    OK = stop!(stp)

    @info log_row(Any[stp.meta.nb_of_stop, "N", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
 
    #update the penalty parameter if necessary
    if !OK
      ncx  = norm(state.cx)
      feas = ncx < norm(stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0), Inf)
      if restoration_phase && !feas &&  stalling >= 3 #or sub_stp.meta.optimal
        #Can"t escape this infeasible stationary point.
        stp.meta.suboptimal = true
        OK = true
      elseif !feas && (stalling >= 3 || unsuccessful_subpb >= 3)
        #we are most likely stuck at an infeasible stationary point.
        restoration_phase = true
        state.x += min(max(stp.meta.atol, 1/σ, 1e-3), 1.) * rand(stp.pb.meta.nvar) 
             
        #Go back to three iterations ago
        σ = max(σ/meta.σ_update^(1.5), meta.σ_0)
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ = max(ρ/meta.ρ_update^(1.5), meta.ρ_0)
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state, x = state.x) #reinitialize the State (keeping x)
             
        stalling = 0
        unsuccessful_subpb = 0
        @info log_row(Any[stp.meta.nb_of_stop, "R", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
      elseif ncx > Δ * nc0 && !feas
        σ *= meta.σ_update
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ *= meta.ρ_update
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
        @info log_row(Any[stp.meta.nb_of_stop, "D", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
      end
      nc0 = copy(ncx)
   end
   
  end #end of main loop
  
  @show status(stp), restoration_phase
  return GenericExecutionStats(status_stopping_to_stats(stp), stp.pb,
                               solution     = stp.current_state.x,
                               objective    = stp.current_state.fx,
                               primal_feas  = norm(stp.current_state.cx, Inf),
                               dual_feas    = sub_stp.current_state.current_score,
                               multipliers  = stp.current_state.lambda,
                               iter         = stp.meta.nb_of_stop,
                               elapsed_time = stp.current_state.current_time - stp.meta.start_time)
end
