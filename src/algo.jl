function fps_solve(stp :: NLPStopping, meta :: AlgoData{T}) where T
  if !(stp.pb.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  if has_inequalities(stp.pb)
    error("Error: consider only problem with equalities and bounds. 
           Use `NLPModelsModifiers.SlackModel`.")
  end
  state = stp.current_state
  #Initialize parameters
  x0, σ, ρ, δ = state.x, meta.σ_0, meta.ρ_0 , meta.δ_0
  
  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(stp.pb, σ, ρ, δ, meta.linear_system_solver, meta.hessian_approx)

  #First call to the stopping
  OK = start!(stp)
  #Prepare the subproblem-stopping for the unconstrained minimization.
  sub_stp = NLPStopping(nlp, NLPAtX(x0), main_stp = stp, 
                                         optimality_check = has_bounds(nlp) ? unconstrained_check : optim_check_bounded,
                                         max_iter = 10000,
                                         atol     = stp.meta.atol,
                                         rtol     = stp.meta.rtol,
                                         unbounded_threshold = 1e20)
  
  nc0 = norm(state.cx, Inf)
  Δ   = 0.95 #expected decrease in feasibility
  unsuccessful_subpb = 0 #number of consecutive failed subproblem solve.
  unbounded_subpb = 0 #number of consecutive failed subproblem solve.
  stalling = 0 #number of consecutive successful subproblem solve without progress
  restoration_phase = false
  
  @info log_header([:iter, :step, :f, :c, :score, :σ, :stat], [Int, String, T, T, T, T, Symbol],
                   hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :σ=>"σ"))
  @info log_row(Any[0, "Init", NaN, nc0, norm(state.current_score,Inf), σ, :Initial])

  while !OK #main loop

    reinit!(sub_stp) #reinit the sub-stopping.
    #Solve the subproblem
    sub_stp = meta.unconstrained_solver(sub_stp)
    
    #Update the State with the info given by the subproblem:
    if sub_stp.meta.optimal || sub_stp.meta.suboptimal
      if sub_stp.current_state.x == state.x
        stalling += 1  
      end
      unsuccessful_subpb, unbounded_subpb = 0, 0
      
      Stopping.update!(state, x      = sub_stp.current_state.x,
                              fx     = sub_stp.pb.fx,
                              gx     = sub_stp.pb.gx,
                              cx     = sub_stp.pb.cx,
                              lambda = sub_stp.pb.ys,
                              res    = grad(sub_stp.pb, sub_stp.current_state.x))

      @info log_row(Any[stp.meta.nb_of_stop, "Optml", 
                     state.fx, norm(state.cx), 
                     sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
   elseif sub_stp.meta.unbounded
      unbounded_subpb += 1
      ncx  = norm(sub_stp.pb.cx)
      feas_tol = stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0)
      feas = ncx < norm(feas_tol, Inf)
      if feas
        stp.meta.unbounded_pb = true #unbounded
      end
      #@show sub_stp.current_state.x, sub_stp.pb.ys, sub_stp.pb.δ
      @info log_row(Any[stp.meta.nb_of_stop, "Unbdd", 
                     sub_stp.current_state.fx, 
                     ncx, 
                     sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
   elseif sub_stp.meta.tired || sub_stp.meta.resources
      unsuccessful_subpb += 1
      @info log_row(Any[stp.meta.nb_of_stop, "Tired", 
                     sub_stp.current_state.fx, 
                     norm(sub_stp.pb.cx),
                     sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
   elseif sub_stp.meta.iteration_limit || sub_stp.meta.stalled
      unsuccessful_subpb += 1
#=
      Stopping.update!(state, x      = sub_stp.current_state.x,
                              fx     = sub_stp.pb.fx,
                              gx     = sub_stp.pb.gx,
                              cx     = sub_stp.pb.cx,
                              lambda = sub_stp.pb.ys,
                              res    = grad(sub_stp.pb, sub_stp.current_state.x))
=#
      @info log_row(Any[stp.meta.nb_of_stop, "Stlld", 
                     state.fx, 
                     norm(state.cx),
                     sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
   else #exception of unexpected failure
      stp.meta.fail_sub_pb = true
      @warn "Exception of unexpected failure: $(status(sub_stp, list = true))"
      #break 
   end
   
    #Check optimality conditions: either stop! is true OR the penalty parameter is too small
    stp.meta.fail_sub_pb = σ > meta.σ_max || ρ > meta.ρ_max #stp.meta.stalled 
    OK = stop!(stp)
 
    #update the penalty parameter if necessary
    if !OK
      ncx  = norm(state.cx) #careful as this is not always updated
      feas_tol = norm(stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0), Inf)
      feas = ncx < feas_tol
      if sub_stp.meta.optimal || sub_stp.meta.suboptimal #we need to tighten the tolerances
        sub_stp.meta.atol /= 10
        sub_stp.meta.rtol /= 10
      elseif restoration_phase && !feas && stalling >= 3 #or sub_stp.meta.optimal
        #Can"t escape this infeasible stationary point.
        stp.meta.suboptimal = true
        OK = true
      elseif unbounded_subpb >= 3 #&& !restoration_phase # && !feas #Tanj: how can we get here???
        # Is that really useful ??????
        restoration_phase = true
        state.x += min(max(stp.meta.atol, 1/σ, 1e-3), 1.) * rand(stp.pb.meta.nvar)   

        #Go back to three iterations ago
        #=
        σ = max(σ/meta.σ_update^(1.5), meta.σ_0)
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ = max(ρ/meta.ρ_update^(1.5), meta.ρ_0)
        =#
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state, x = state.x) #reinitialize the State (keeping x)
             
        unbounded_subpb = 0
        @info log_row(Any[stp.meta.nb_of_stop, "R-Unbdd", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)]) #why state.fx if we just reinit it?
      #=
      elseif sub_stp.meta.unbounded #unbounded subproblem no restoration
        σ *= meta.σ_update
        sub_stp.pb.σ = σ
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
        @info log_row(Any[stp.meta.nb_of_stop, "noR-Unbdd", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
      =#
      elseif (stalling >= 3 || unsuccessful_subpb >= 3) && !feas
        #we are most likely stuck at an infeasible stationary point.
        #or an undetected unbounded problem
        restoration_phase = true
        ##################################################################
        #
        # Add a restoration step here
        #
        ρ = feas_tol #by default, we just want a feasible point.
        Jx = jac_op(stp.pb, state.x)
        z, cz, normcz, Jz, status_feas = feasibility_step(stp.pb, state.x, state.cx, ncx, Jx, ρ, feas_tol)
        if status_feas == :success
          Stopping.update!(stp.current_state, x = z, cx = cz)
        else
          #randomization step:
          state.x += min(max(stp.meta.atol, 1/σ, 1e-3), 1.) * rand(stp.pb.meta.nvar)   
        end
        #
        # End of restoration step
        #
        ###################################################################
        #Go back to three iterations ago
        σ = max(σ/meta.σ_update^(1.5), meta.σ_0)
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ = max(ρ/meta.ρ_update^(1.5), meta.ρ_0)
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state, x = state.x) #reinitialize the State (keeping x)
             
        stalling, unsuccessful_subpb = 0, 0
        @info log_row(Any[stp.meta.nb_of_stop, "R", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)]) #why state.fx if we just reinit it?
      elseif ncx > Δ * nc0 && !feas
        σ *= meta.σ_update
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ *= meta.ρ_update
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
        @info log_row(Any[stp.meta.nb_of_stop, "D", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
      elseif stalling >= 3 || unsuccessful_subpb >= 3 #but feas is true
        #probably, an undetected unbounded problem
        σ *= meta.σ_update
        sub_stp.pb.σ = σ
        sub_stp.pb.ρ *= meta.ρ_update
        #reinitialize the State(s) as the problem changed
        reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
        @info log_row(Any[stp.meta.nb_of_stop, "F-D", 
                     state.fx, norm(state.cx), sub_stp.current_state.current_score, 
                     σ, status(sub_stp)])
      else
        @show "Euh... How?", stalling, unsuccessful_subpb, unbounded_subpb, sub_stp.meta.unbounded, feas
        # ("Euh... How?", 0, 2, 0, false, false) -> and then R steps which never increase sigma
      end
      nc0 = copy(ncx)
   end
   
  end #end of main loop
  
#  @show status(stp), restoration_phase
  return GenericExecutionStats(status_stopping_to_stats(stp), stp.pb,
                               solution     = stp.current_state.x,
                               objective    = stp.current_state.fx,
                               primal_feas  = norm(stp.current_state.cx, Inf),
                               dual_feas    = sub_stp.current_state.current_score,
                               multipliers  = stp.current_state.lambda,
                               iter         = stp.meta.nb_of_stop,
                               elapsed_time = stp.current_state.current_time - stp.meta.start_time,
                               solver_specific = Dict(
                                                 :stp => stp,
                                                 :restoration => restoration_phase
                                                 ))
end
