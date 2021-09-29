#Tangi: same implementation as in DCI.jl
"""    feasibility_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

Given xₖ, finds min ‖cₖ + Jₖd‖
"""
function feasibility_step(
  feasibility_solver::GNSolver,
  nlp::AbstractNLPModel,
  x::AbstractVector{T},
  cx::AbstractVector{T},
  normcx::T,
  Jx::Union{LinearOperator{T}, AbstractMatrix{T}},
  ρ::T,
  ctol::AbstractFloat;
  η₁::AbstractFloat = 1e-3,
  η₂::AbstractFloat = 0.66,
  σ₁::AbstractFloat = 0.25,
  σ₂::AbstractFloat = 2.0,
  Δ0::T = one(T),
  max_eval::Int = 1_000,
  max_time::AbstractFloat = 60.0,
  max_feas_iter::Int = typemax(Int64),
) where {T, S}
  z = x
  cz = cx
  Jz = Jx
  zp = feasibility_solver.workspace_zp
  czp = feasibility_solver.workspace_czp
  Jd = feasibility_solver.workspace_Jd
  normcz = normcx # cons(nlp, x) = normcx = normcz for the first z

  Δ = Δ0

  feas_iter = 0
  consecutive_bad_steps = 0 # Bad steps are when ‖c(z)‖ / ‖c(x)‖ > 0.95
  failed_step_comp = false

  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  status = :unknown

  start_time = time()
  infeasible = false

  while !(normcz ≤ ρ || tired || infeasible)

    #Compute the a direction satisfying the trust-region constraint
    d, Jd, infeasible, solved =
      TR_lsmr(feasibility_solver.TR_compute_step, cz, Jz, ctol, Δ, normcz, Jd)

    if infeasible #the direction is too small
      failed_step_comp = true #too small step
      status = :too_small
    else
      zp = z + d
      czp = cons(nlp, zp)
      normczp = norm(czp)

      Pred = T(0.5) * (normcz^2 - norm(Jd + cz)^2)
      Ared = T(0.5) * (normcz^2 - normczp^2)

      if Ared / Pred < η₁
        Δ = max(T(1e-8), Δ * σ₁)
        status = :reduce_Δ
      else #success
        z = zp
        Jz = jac_op!(nlp, z, feasibility_solver.workspace_Jv, feasibility_solver.workspace_Jtv)
        cz = czp
        if normczp / normcz > T(0.95)
          consecutive_bad_steps += 1
        else
          consecutive_bad_steps = 0
        end
        normcz = normczp
        status = :success
        if Ared / Pred > η₂ && norm(d) >= T(0.99) * Δ
          Δ *= σ₂
        end
      end
    end

    @info log_row(
      Any[
        "F",
        feas_iter,
        neval_obj(nlp) + neval_cons(nlp),
        Float64,
        Float64,
        Float64,
        normcz,
        Float64,
        Float64,
        status,
        norm(d),
        Δ,
      ],
    )

    # Safeguard: aggressive normal step
    if normcz > ρ && (consecutive_bad_steps ≥ feasibility_solver.bad_steps_lim || failed_step_comp)
      Hz = hess_op(nlp, z, cz, obj_weight = zero(T))
      Krylov.solve!(feasibility_solver.aggressive_step, Hz + Jz' * Jz, Jz' * cz)
      d = feasibility_solver.aggressive_step.x
      stats = feasibility_solver.aggressive_step.stats
      if !stats.solved
        @warn "Fail cg in feasibility_step: $(stats.status)"
      end
      @. zp = z - d
      cons!(nlp, zp, czp)
      nczp = norm(czp)
      if nczp < normcz #even if d is small we keep going
        infeasible = false
        failed_step_comp = false
        status = :aggressive
        z, cz = zp, czp
        normcz = nczp
        Jz = jac_op!(nlp, z, feasibility_solver.workspace_Jv, feasibility_solver.workspace_Jtv)
      elseif norm(d) < ctol * min(nczp, one(T))
        infeasible = true
        status = :aggressive_fail
      else #unsuccessful,nczp > normcz,infeasible = true,status = :too_small
        cg_iter = length(stats.residuals)
        #@show cg_iter, stats.residuals[end], nczp, normcz, norm(Jz' * czp)
        #should we increase the iteration limit if we busted it?
        #Adding regularization might be more efficient
      end
      @info log_row(
        Any[
          "F-safe",
          feas_iter,
          neval_obj(nlp) + neval_cons(nlp),
          Float64,
          Float64,
          Float64,
          normcz,
          Float64,
          Float64,
          status,
          norm(d),
          Δ,
        ],
      )
    end

    el_time = time() - start_time
    feas_iter += 1
    many_evals = neval_obj(nlp) + neval_cons(nlp) > max_eval
    iter_limit = feas_iter > max_feas_iter
    tired = many_evals || el_time > max_time || iter_limit
  end

  status = if normcz ≤ ρ
    :success
  elseif tired
    if neval_obj(nlp) + neval_cons(nlp) > max_eval
      :max_eval
    elseif el_time > max_time
      :max_time
    elseif feas_iter > max_feas_iter
      :max_iter
    else
      :unknown_tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return z, cz, normcz, Jz, status
end

function TR_lsmr(
  solver,
  cz::AbstractVector{T},
  Jz::Union{LinearOperator{T}, AbstractMatrix{T}},
  ctol::AbstractFloat,
  Δ::T,
  normcz::AbstractFloat,
  Jd::AbstractVector{T},
) where {T}
  Krylov.solve!(solver, Jz, -cz, radius = Δ)
  d = solver.x
  stats = solver.stats

  infeasible = norm(d) < ctol * min(normcz, one(T))
  solved = stats.solved
  if !solved
    @warn "Fail lsmr in TR_lsmr: $(stats.status)"
  end

  Jd .= Jz * d #lsmr doesn't return this information

  return d, Jd, infeasible, solved
end
