function fps_solve(
  stp::NLPStopping,
  fpssolver::FPSSSolver{T, QDS, US};
  subsolver_verbose::Int = 0,
  subsolver_max_iter::Int = 20000,
  lagrange_bound::T = 1 / sqrt(eps(T)),
) where {T, QDS, US}
  meta = fpssolver.meta
  feasibility_solver = fpssolver.feasibility_solver
  if !(stp.pb.meta.minimize)
    error("fps_solve only works for minimization problem")
  end
  if has_inequalities(stp.pb)
    error("Error: consider only problem with equalities and bounds. 
           Use `NLPModelsModifiers.SlackModel`.")
  end
  state = stp.current_state
  #Initialize parameters
  σ, ρ, δ = meta.σ_0, meta.ρ_0, zero(T)

  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(
    stp.pb,
    σ,
    ρ,
    δ,
    meta.hessian_approx,
    explicit_linear_constraints = meta.explicit_linear_constraints,
    qds = fpssolver.qdsolver,
  )

  #First call to the stopping
  OK = if (get_ncon(stp.pb) > 0) & (stp.current_state.cx == [])
    update_and_start!(stp, lambda = stp.pb.meta.y0, cx = cons(stp.pb, stp.current_state.x)) # , convert = true)
  else
    start!(stp)
  end
  #Prepare the subproblem-stopping for the subproblem minimization.
  sub_state = if nlp.meta.ncon > 0
    NLPAtX(state.x, nlp.meta.y0, Jx = jac(nlp, state.x), res = zeros(T, nlp.meta.nvar))
  else
    NLPAtX(state.x, res = zeros(T, nlp.meta.nvar))
  end
  sub_stp = NLPStopping(
    nlp,
    sub_state,
    main_stp = stp,
    optimality_check = if nlp.meta.ncon > 0
      KKT
    elseif has_bounds(nlp)
      optim_check_bounded
    else
      unconstrained_check
    end,
    max_iter = subsolver_max_iter,
    atol = meta.atol_sub(stp.meta.atol), # max(0.1, stp.meta.atol),# stp.meta.atol / 100,
    rtol = meta.rtol_sub(stp.meta.rtol), # max(0.1, stp.meta.rtol), #stp.meta.rtol / 100,
    optimality0 = one(T),
    unbounded_threshold = meta.subpb_unbounded_threshold,
  )

  nc0 = norm(state.cx, Inf)
  feas_tol = stp.meta.atol # norm(stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0), Inf)
  Δ = meta.Δ # 0.95 #expected decrease in feasibility
  unsuccessful_subpb = 0 #number of consecutive failed subproblem solve.
  unbounded_subpb = 0 #number of consecutive failed subproblem solve.
  stalling = 0 #number of consecutive successful subproblem solve without progress
  feasibility_phase, restoration_phase = false, false

  @info log_header(
    [:iter, :step, :f, :c, :score, :σ, :ρ, :δ, :stat, :η, :λ],
    [Int, String, T, T, T, T, T, T, Symbol, T, T],
    hdr_override = Dict(
      :f => "f(x)",
      :c => "‖c(x)‖",
      :score => "‖∇L‖",
      :σ => "σ",
      :ρ => "ρ",
      :δ => "δ",
      :η => "η",
      :λ => "‖λ‖",
    ),
  )
  @info log_row(
    Any[
      0,
      "Init",
      NaN,
      nc0,
      norm(state.current_score, Inf),
      σ,
      ρ,
      δ,
      :Initial,
      sub_stp.pb.η,
      NaN, # sub_stp.pb.ys is undef
    ],
  )

  while !OK
    reinit!(sub_stp) #reinit the sub-stopping.
    #Solve the subproblem
    sub_stp = meta.subproblem_solver(sub_stp, subsolver_verbose = subsolver_verbose)

    unbounded_lagrange_multiplier = norm(sub_stp.pb.ys, Inf) ≥ lagrange_bound # add to stopping meta?

    #Update the State with the info given by the subproblem:
    if sub_stp.meta.optimal || sub_stp.meta.suboptimal
      if sub_stp.current_state.x == state.x
        stalling += 1
      end
      unsuccessful_subpb, unbounded_subpb = 0, 0

      if stp.pb.meta.nlin > 0 && nlp.explicit_linear_constraints
        stp.current_state.lambda[stp.pb.meta.lin] .= -sub_stp.current_state.lambda
        stp.current_state.lambda[stp.pb.meta.nln] .= -sub_stp.pb.ys
        stp.current_state.cx[stp.pb.meta.lin] .= sub_stp.current_state.cx
        stp.current_state.cx[stp.pb.meta.nln] .= sub_stp.pb.cx .+ get_lcon(stp.pb)[stp.pb.meta.nln]
        Stopping.update!(
          state,
          res = sub_stp.current_state.gx + sub_stp.current_state.Jx' * sub_stp.current_state.lambda,
        )
      else
        stp.current_state.lambda .= -sub_stp.pb.ys
        stp.current_state.cx .= sub_stp.pb.cx + get_lcon(stp.pb)
        Stopping.update!(state, res = sub_stp.current_state.gx)
      end
      Stopping.update!(
        state,
        x = sub_stp.current_state.x,
        fx = sub_stp.pb.fx,
        gx = sub_stp.pb.gx,
        Jx = jac(stp.pb, sub_stp.current_state.x),
        mu = sub_stp.current_state.mu,
        convert = true,
      )
      go_log(stp, sub_stp, state.fx, norm(sub_stp.pb.cx), "Optml")
    elseif sub_stp.meta.unbounded || sub_stp.meta.unbounded_pb || unbounded_lagrange_multiplier
      stalling, unsuccessful_subpb = 0, 0
      unbounded_subpb += 1
      ncx = norm(sub_stp.pb.cx, Inf) # norm(sub_stp.pb.cx, Inf)
      feas = ncx < norm(feas_tol, Inf)
      if feas
        stp.meta.unbounded_pb = true
      end
      go_log(stp, sub_stp, sub_stp.current_state.fx, ncx, "Unbdd")
    elseif sub_stp.meta.tired || sub_stp.meta.resources
      stalling, unbounded_subpb = 0, 0
      unsuccessful_subpb += 1
      # ncx = 
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
            ncx = norm(state.cx)
      =#
      ncx = norm(sub_stp.pb.cx)
      go_log(stp, sub_stp, state.fx, ncx, "Stlld")
    else #exception of unexpected failure
      stp.meta.fail_sub_pb = true
      @warn "Exception of unexpected failure: $(status(sub_stp, list = true))"
      #break 
    end

    #Check optimality conditions: either stop! is true OR the penalty parameter is too small
    stp.meta.fail_sub_pb =
      stp.meta.fail_sub_pb || (nlp.σ > meta.σ_max || nlp.ρ > meta.ρ_max || nlp.δ > meta.δ_max)
    OK = stop!(stp)

    if !OK
      ncx = norm(sub_stp.pb.cx) # state.cx is updated in optimal cases only
      feas = ncx < feas_tol
      if (sub_stp.meta.optimal || sub_stp.meta.suboptimal)
        if feas #we need to tighten the tolerances
          sub_stp.meta.atol = max(sub_stp.meta.atol / 10, eps(T)) # put in parameters
          sub_stp.meta.rtol = max(sub_stp.meta.rtol / 10, eps(T)) # put in parameters
          # If we reach the tolerence limit here, the tol tighten - stalling
          sub_stp.pb.η = max(meta.η_1, sub_stp.pb.η * meta.η_update)
          sub_stp.pb.xk .= state.x
          go_log(stp, sub_stp, state.fx, ncx, "D-ϵ")
        elseif !feasibility_phase && (stalling ≥ 3 || sub_stp.meta.atol < eps(T))
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unbounded_subpb = 0
          restoration_feasibility!(feasibility_solver, meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx - get_lcon(stp.pb)), "R")
        elseif stalling ≥ 3 || sub_stp.meta.atol < eps(T)
          # infeasible stationary point
          stp.meta.suboptimal = true
          OK = true
        else
          # update parameters to increase feasibility
          update_parameters!(meta, sub_stp, feas)
          go_log(stp, sub_stp, state.fx, ncx, "D")
        end
      elseif sub_stp.meta.unbounded || sub_stp.meta.unbounded_pb || unbounded_lagrange_multiplier
        if !feasibility_phase && unbounded_subpb ≥ 3 && !feas
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unbounded_subpb = 0
          restoration_feasibility!(feasibility_solver, meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx - get_lcon(stp.pb)), "R")
        elseif !restoration_phase && unbounded_subpb ≥ 3
          restoration_phase = true
          unbounded_subpb = 0
          random_restoration!(meta, stp, sub_stp)
          go_log(stp, sub_stp, state.fx, norm(state.cx - get_lcon(stp.pb)), "R-Unbdd")
        else
          # update parameters to increase feasibility
          update_parameters_unbdd!(meta, sub_stp, feas)
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
          go_log(stp, sub_stp, state.fx, norm(state.cx - get_lcon(stp.pb)), "R-Unscc")
        elseif !feasibility_phase && unsuccessful_subpb ≥ 3 && !feas
          #we are most likely stuck at an infeasible stationary point.
          #or an undetected unbounded problem
          feasibility_phase = true
          unsuccessful_subpb = 0
          restoration_feasibility!(feasibility_solver, meta, stp, sub_stp, feas_tol, ncx)

          stalling, unsuccessful_subpb = 0, 0
          go_log(stp, sub_stp, state.fx, norm(state.cx - get_lcon(stp.pb)), "R")
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
    primal_feas = norm(stp.current_state.cx - get_lcon(stp.pb), Inf),
    dual_feas = sub_stp.current_state.current_score,
    multipliers = stp.current_state.lambda,
    multipliers_L = stp.current_state.mu,
    iter = stp.meta.nb_of_stop,
    elapsed_time = stp.current_state.current_time - stp.meta.start_time,
    # solver_specific = Dict(:stp => stp, :restoration => restoration_phase),
  )
end

"""
    restoration_feasibility!(feasibility_solver, meta, stp, sub_stp, feas_tol, ncx)

Try to find a feasible point, see [`feasibility_step`](@ref).
"""
function restoration_feasibility!(feasibility_solver, meta, stp, sub_stp, feas_tol, ncx)
  # by default, we just want a feasible point
  ϵ_feas = feas_tol
  Jx = jac(stp.pb, stp.current_state.x)
  z, cz, normcz, Jz, status_feas = feasibility_step(
    feasibility_solver,
    stp.pb,
    stp.current_state.x,
    stp.current_state.cx - get_lcon(stp.pb),
    ncx,
    Jx,
    ϵ_feas,
    feas_tol,
  )
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
  reinit!(
    sub_stp.current_state,
    x = stp.current_state.x,
    cx = sub_stp.current_state.cx,
    Jx = sub_stp.current_state.Jx,
    res = sub_stp.current_state.res,
  ) #reinitialize the State (keeping x, cx, Jx)
  # Should we also update the stp.current_state ??
end

"""
    random_restoration!(meta, stp, sub_stp)

Add a random perturbation to the current iterate.
"""
function random_restoration!(meta, stp, sub_stp)
  σ = sub_stp.pb.σ
  radius = min(max(stp.meta.atol, 1 / σ, 1e-3), 1.0)
  stp.current_state.x .+= radius * rand(stp.pb.meta.nvar)

  # Go back to three iterations ago
  # σ = max(σ/meta.σ_update^(1.5), meta.σ_0)
  # sub_stp.pb.σ = σ
  # sub_stp.pb.ρ = max(ρ/meta.ρ_update^(1.5), meta.ρ_0)

  # reinitialize the State(s) as the problem changed
  reinit!(
    sub_stp.current_state,
    x = stp.current_state.x,
    cx = sub_stp.current_state.cx,
    Jx = sub_stp.current_state.Jx,
    res = sub_stp.current_state.res,
  ) #reinitialize the State (keeping x, cx, Jx)
  # Should we also update the stp.state ??
end

"""
    update_parameters!(meta, sub_stp, feas)

Update `σ`. If the current iterate also update `ρ`.
"""
function update_parameters!(meta, sub_stp, feas)
  # update parameters to increase feasibility
  # Do something different if ncx > Δ * nc0 ?
  sub_stp.pb.σ *= meta.σ_update
  if !feas # Do we really come here with a feasible point ?
    sub_stp.pb.ρ *= meta.ρ_update
  end
  #reinitialize the State(s) as the problem changed
  reinit!(
    sub_stp.current_state,
    cx = sub_stp.current_state.cx,
    Jx = sub_stp.current_state.Jx,
    res = sub_stp.current_state.res,
  ) #reinitialize the state (keeping x, cx, Jx)
end

"""
    update_parameters_unbdd!(meta, sub_stp, feas)

Start or update `δ`, then call [`update_parameters!(meta, sub_stp, feas)`](@ref)
"""
function update_parameters_unbdd!(meta, sub_stp, feas)
  if sub_stp.pb.δ == 0
    sub_stp.pb.δ = meta.δ_0
  else
    sub_stp.pb.δ *= meta.δ_update
  end
  update_parameters!(meta, sub_stp, feas)
end

"""
    go_log(stp, sub_stp, fx, ncx, mess::String)

Logging shortcut.
"""
function go_log(stp, sub_stp, fx, ncx, mess::String)
  @info log_row(
    Any[
      stp.meta.nb_of_stop,
      mess,
      fx,
      ncx,
      sub_stp.current_state.current_score,
      sub_stp.pb.σ,
      sub_stp.pb.ρ,
      sub_stp.pb.δ,
      status(sub_stp),
      sub_stp.pb.η,
      norm(sub_stp.pb.ys, Inf),
    ],
  )
end
