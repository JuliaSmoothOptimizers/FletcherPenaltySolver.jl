function fps_solve(stp::NLPStopping, fpssolver::FPSSSolver{T, QDS, US}) where {T, QDS, US}
  meta = fpssolver.meta
  if !(stp.pb.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  if has_inequalities(stp.pb)
    error("Error: consider only problem with equalities and bounds. 
           Use `NLPModelsModifiers.SlackModel`.")
  end
  state = stp.current_state
  #Initialize parameters
  σ, ρ, δ = meta.σ_0, meta.ρ_0, meta.δ_0

  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(stp.pb, σ, ρ, δ, meta.hessian_approx, qds = fpssolver.qdsolver)

  #First call to the stopping
  OK = start!(stp)
  #Prepare the subproblem-stopping for the unconstrained minimization.
  sub_stp = NLPStopping(
    nlp,
    NLPAtX(state.x),
    main_stp = stp,
    optimality_check = has_bounds(nlp) ? unconstrained_check : optim_check_bounded,
    max_iter = 10000,
    atol = meta.atol_sub(stp.meta.atol), # max(0.1, stp.meta.atol),# stp.meta.atol / 100,
    rtol = meta.rtol_sub(stp.meta.rtol), # max(0.1, stp.meta.rtol), #stp.meta.rtol / 100,
    unbounded_threshold = meta.subpb_unbounded_threshold,
  )

  nc0 = norm(state.cx, Inf)
  Δ = meta.Δ # 0.95 #expected decrease in feasibility
  unsuccessful_subpb = 0 #number of consecutive failed subproblem solve.
  unbounded_subpb = 0 #number of consecutive failed subproblem solve.
  stalling = 0 #number of consecutive successful subproblem solve without progress
  feasibility_phase, restoration_phase = false, false

  @info log_header(
    [:iter, :step, :f, :c, :score, :σ, :stat, :η],
    [Int, String, T, T, T, T, Symbol],
    hdr_override = Dict(:f => "f(x)", :c => "||c(x)||", :score => "‖∇L‖", :σ => "σ", :η => "η"),
  )
  @info log_row(Any[0, "Init", NaN, nc0, norm(state.current_score, Inf), σ, :Initial])

  while !OK
    reinit!(sub_stp) #reinit the sub-stopping.
    #Solve the subproblem
    sub_stp = meta.unconstrained_solver(sub_stp)

    #Update the State with the info given by the subproblem:
    if sub_stp.meta.optimal || sub_stp.meta.suboptimal
      if sub_stp.current_state.x == state.x
        stalling += 1
      end
      unsuccessful_subpb, unbounded_subpb = 0, 0

      Stopping.update!(
        state,
        x = sub_stp.current_state.x,
        fx = sub_stp.pb.fx,
        gx = sub_stp.pb.gx,
        cx = sub_stp.pb.cx,
        lambda = sub_stp.pb.ys,
        mu = sub_stp.current_state.mu,
        res = sub_stp.current_state.gx, # grad(sub_stp.pb, sub_stp.current_state.x), # Shouldn't this be returned by the solver?
      )
      go_log(stp, sub_stp, state.fx, norm(state.cx), "Optml")
    elseif sub_stp.meta.unbounded || sub_stp.meta.unbounded_pb
      stalling, unsuccessful_subpb = 0, 0
      unbounded_subpb += 1
      ncx = norm(sub_stp.pb.cx, Inf)
      feas_tol = stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0)
      feas = ncx < norm(feas_tol, Inf)
      if feas
        stp.meta.unbounded_pb = true
      end
      go_log(stp, sub_stp, sub_stp.current_state.fx, ncx, "Unbdd")
    elseif sub_stp.meta.tired || sub_stp.meta.resources
      stalling, unbounded_subpb = 0, 0
      unsuccessful_subpb += 1
      go_log(stp, sub_stp, sub_stp.current_state.fx, norm(sub_stp.pb.cx), "Tired")
    elseif sub_stp.meta.iteration_limit || sub_stp.meta.stalled
      stalling, unbounded_subpb = 0, 0
      unsuccessful_subpb += 1
      #=
            Stopping.update!(state, x      = sub_stp.current_state.x,
                                    fx     = sub_stp.pb.fx,
                                    gx     = sub_stp.pb.gx,
                                    cx     = sub_stp.pb.cx,
                                    lambda = sub_stp.pb.ys,
                                    res    = grad(sub_stp.pb, sub_stp.current_state.x))
      =#
      go_log(stp, sub_stp, state.fx, norm(state.cx), "Stlld")
    else #exception of unexpected failure
      stp.meta.fail_sub_pb = true
      @warn "Exception of unexpected failure: $(status(sub_stp, list = true))"
      #break 
    end

    #Check optimality conditions: either stop! is true OR the penalty parameter is too small
    stp.meta.fail_sub_pb = stp.meta.fail_sub_pb || (nlp.σ > meta.σ_max || nlp.ρ > meta.ρ_max)
    OK = stop!(stp)

    if !OK
      ncx = norm(state.cx) #!!! careful as this is not always updated
      feas_tol = norm(stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0), Inf)
      feas = ncx < feas_tol
      if (sub_stp.meta.optimal || sub_stp.meta.suboptimal)
        if feas #we need to tighten the tolerances
          sub_stp.meta.atol /= 10 # put in parameters
          sub_stp.meta.rtol /= 10 # put in parameters
          sub_stp.pb.η = max(meta.η_1, sub_stp.pb.η * meta.η_update)
          sub_stp.pb.xk .= state.x
          go_log(stp, sub_stp, state.fx, ncx, "D-ϵ")
        elseif !feasibility_phase && (stalling ≥ 3 || sub_stp.meta.atol < eps(T))
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unbounded_subpb = 0
          restoration_feasibility!(meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx), "R")
        elseif stalling ≥ 3 || sub_stp.meta.atol < eps(T)
          # infeasible stationary point
          stp.meta.suboptimal = true
          Ok = true
        else
          # update parameters to increase feasibility
          update_parameters!(meta, sub_stp, feas)
          go_log(stp, sub_stp, state.fx, ncx, "D")
        end
      elseif sub_stp.meta.unbounded || sub_stp.meta.unbounded_pb
        if !feasibility_phase && unbounded_subpb ≥ 3 && !feas
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unbounded_subpb = 0
          restoration_feasibility!(meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx), "R")
        elseif !restoration_phase && unbounded_subpb ≥ 3
          restoration_phase = true
          unbounded_subpb = 0
          random_restoration!(meta, stp, sub_stp)
          go_log(stp, sub_stp, state.fx, norm(state.cx), "R-Unbdd")
        else
          # update parameters to increase feasibility
          update_parameters!(meta, sub_stp, feas)
          go_log(stp, sub_stp, state.fx, ncx, "D")
        end
      elseif sub_stp.meta.tired ||
             sub_stp.meta.resources ||
             sub_stp.meta.iteration_limit ||
             sub_stp.meta.stalled
        if !restoration_phase && unsuccessful_subpb ≥ 3 && feas
          restoration_phase = true
          unsuccessful_subpb = 0
          random_restoration!(meta, stp, sub_stp)
          go_log(stp, sub_stp, state.fx, norm(state.cx), "R-Unscc")
        elseif !feasibility_phase && unsuccessful_subpb ≥ 3 && !feas
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unsuccessful_subpb = 0
          restoration_feasibility!(meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx), "R")
        else
          # update parameters to increase feasibility
          update_parameters!(meta, sub_stp, feas)
          go_log(stp, sub_stp, state.fx, ncx, "D")
        end
      else
        @show "Euh... How?", stp.pb.meta.name
        stalling, unsuccessful_subpb, unbounded_subpb, sub_stp.meta.unbounded, feas
      end
      nc0 = copy(ncx)
    end
  end #end of main loop

  return GenericExecutionStats(
    status_stopping_to_stats(stp),
    stp.pb,
    solution = stp.current_state.x,
    objective = stp.current_state.fx,
    primal_feas = norm(stp.current_state.cx, Inf),
    dual_feas = sub_stp.current_state.current_score,
    multipliers = stp.current_state.lambda,
    multipliers_L = stp.current_state.mu,
    iter = stp.meta.nb_of_stop,
    elapsed_time = stp.current_state.current_time - stp.meta.start_time,
    # solver_specific = Dict(:stp => stp, :restoration => restoration_phase),
  )
end

function restoration_feasibility!(meta, stp, sub_stp, feas_tol, ncx)
  # by default, we just want a feasible point
  ϵ_feas = feas_tol
  Jx = jac_op(stp.pb, stp.current_state.x)
  z, cz, normcz, Jz, status_feas =
    feasibility_step(stp.pb, stp.current_state.x, stp.current_state.cx, ncx, Jx, ϵ_feas, feas_tol)
  if status_feas == :success
    Stopping.update!(stp.current_state, x = z, cx = cz)
  else
    #randomization step:
    σ = sub_stp.pb.σ
    radius = min(max(stp.meta.atol, 1 / σ, 1e-3), 1.0)
    stp.current_state.x .+= radius * rand(stp.pb.meta.nvar)
  end

  # Go back to three iterations ago
  # σ = max(σ / meta.σ_update^(1.5), meta.σ_0)
  # sub_stp.pb.σ = σ
  # sub_stp.pb.ρ = max(ρ / meta.ρ_update^(1.5), meta.ρ_0)

  # reinitialize the State(s) as the problem changed
  reinit!(sub_stp.current_state, x = stp.current_state.x) #reinitialize the State (keeping x)
  # Should we also update the stp.current_state ??
end

function random_restoration!(meta, stp, sub_stp)
  σ = sub_stp.pb.σ
  radius = min(max(stp.meta.atol, 1 / σ, 1e-3), 1.0)
  stp.current_state.x .+= radius * rand(stp.pb.meta.nvar)

  # Go back to three iterations ago
  # σ = max(σ/meta.σ_update^(1.5), meta.σ_0)
  # sub_stp.pb.σ = σ
  # sub_stp.pb.ρ = max(ρ/meta.ρ_update^(1.5), meta.ρ_0)

  # reinitialize the State(s) as the problem changed
  reinit!(sub_stp.current_state, x = stp.current_state.x) #reinitialize the State (keeping x)
  # Should we also update the stp.state ??
end

function update_parameters!(meta, sub_stp, feas)
  # update parameters to increase feasibility
  # Do something different if ncx > Δ * nc0 ?
  sub_stp.pb.σ *= meta.σ_update
  if !feas # Do we really come here with a feasible point ?
    sub_stp.pb.ρ *= meta.ρ_update
  end
  #reinitialize the State(s) as the problem changed
  reinit!(sub_stp.current_state) #reinitialize the state (keeping x)
end

function go_log(stp, sub_stp, fx, ncx, mess::String)
  @info log_row(
    Any[
      stp.meta.nb_of_stop,
      mess,
      fx,
      ncx,
      sub_stp.current_state.current_score,
      sub_stp.pb.σ,
      status(sub_stp),
      sub_stp.pb.η,
    ],
  )
end
