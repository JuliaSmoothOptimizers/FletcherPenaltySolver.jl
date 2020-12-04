function Fletcher_penalty_optimality_check(pb :: AbstractNLPModel, state :: NLPAtX)
    #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
    #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \sigma  (x 0 )\|  \infty  )
    #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \sigma  (x 0 )\|  \infty  )
    # returns i) + ii) OR iii) ?
    nxk, nlk = norm(state.x, Inf), norm(state.lambda, Inf)
    ϵ = 1e-8
    cx  = state.cx  .- ϵ * max(nxk, 1.)
    gx  = state.gx  .- ϵ * max(nlk, 1.)
    res = state.res .- ϵ * max(nlk, 1.)
 return vcat(cx, res)
end

mutable struct AlgoData{T <: Number}
    σ_min    :: Number#    = eps(T),
    σ_update :: Number#    = T(0.95),
end

function AlgoData(T         :: Number;
                   σ_min    :: Number = eps(T),
                   σ_update :: Number = T(0.95))
   return AlgoData(σ_min, σ_update)
end
include("lbfgs.jl")

export Fletcher_penalty_solver
"""
Solver for equality constrained non-linear programs based on Fletcher's penalty function.

    Cite: Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
    Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
    SIAM Journal on Scientific Computing, 42(3), A1809-A1835.

`Fletcher_penalty_solver(:: NLPStopping, :: AbstractVector{T};  σ_0 :: Number = one(T), σ_min :: Number = eps(T), σ_update :: Number = T(0.4), linear_system_solver :: Function  = _solve_with_linear_operator, unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`
or
`Fletcher_penalty_solver(:: AbstractNLPModel, :: AbstractVector{T}, σ_0 :: Number = one(T), σ_min :: Number = eps(T), σ_update :: Number = T(0.4), linear_system_solver :: Function = _solve_with_linear_operator, unconstrained_solver :: Function = lbfgs) where T <: AbstractFloat`

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
                                 σ_0                   :: Number    = 1.,
                                 σ_min                 :: Number    = eps(),
                                 σ_update              :: Number    = 0.95,
                                 linear_system_solver  :: Function  = _solve_with_linear_operator,
                                 unconstrained_solver  :: Function  = lbfgs,
                                 kwargs...)

 cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
 #Tanj: how to handle stopping criteria where tol_check depends on the State?
 Fptc(atol, rtol, opt0) = rtol * vcat(ones(nlp.meta.ncon) .+ norm(cx0, Inf),
                                      ones(nlp.meta.nvar) .+ norm(gx0, Inf))
                                      
 initial_state = NLPAtX(x0, zeros(nlp.meta.ncon), cx = cx0, gx = gx0, res = gx0)
 stp = NLPStopping(nlp, initial_state,
                   optimality_check = Fletcher_penalty_optimality_check,
                   rtol = 1e-8,
                   tol_check = Fptc,
                   max_cntrs = _init_max_counters(quick = typemax(Int64)); kwargs...)

 return Fletcher_penalty_solver(stp,
                                σ_0 = σ_0, σ_min = σ_min, σ_update = σ_update,
                                linear_system_solver = linear_system_solver,
                                unconstrained_solver = unconstrained_solver)
end

function Fletcher_penalty_solver(stp                   :: NLPStopping;
                                 σ_0                   :: Number    = one(),
                                 σ_min                 :: Number    = eps(),
                                 σ_update              :: Number    = 0.95,
                                 linear_system_solver  :: Function  = _solve_with_linear_operator,
                                 unconstrained_solver  :: Function  = lbfgs)
  state = stp.current_state
  #Initialize parameters
  x0, σ = state.x, σ_0
  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(stp.pb, σ_0, linear_system_solver)

  #First call to the stopping
  OK = start!(stp)
  #Prepare the subproblem-stopping for the unconstrained minimization.
  sub_stp = NLPStopping(nlp, NLPAtX(x0), main_stp = stp, optimality_check = unconstrained_check)

  T = eltype(x0)
  @info log_header([:iter, :f, :c, :score, :sigma], [Int, T, T, T, T],
                   hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :sigma=>"σ"))
  @info log_row(Any[0, NaN, norm(state.cx,Inf), norm(state.current_score,Inf), σ])

  while !OK #main loop

   #Solve the subproblem
   reinit!(sub_stp) #reinit the sub-stopping.
   sub_stp = unconstrained_solver(sub_stp)
   #Update the State with the info given by the subproblem:
   if !sub_stp.meta.fail_sub_pb
      Stopping.update!(state, x      = sub_stp.current_state.x,
                              fx     = sub_stp.pb.fx,
                              gx     = sub_stp.pb.gx,
                              cx     = sub_stp.pb.cx,
                              lambda = sub_stp.pb.ys,
                              res    = sub_stp.current_state.gx) #State lacks a bit of flexibility (here sub_stp.current_state.fx and sub_stp.current_state.Hx are lost)
   else
       stp.meta.fail_sub_pb = true
   end
   
   #Check optimality conditions: either stop! is true OR the penalty parameter is too small
   if σ < σ_min stp.meta.fail_sub_pb = true end #stp.meta.stalled
   OK = stop!(stp)

   @info log_row(Any[stp.meta.nb_of_stop, state.fx, norm(state.cx,Inf), norm(state.current_score,Inf), σ])

   #update the penalty parameter if necessary
   if !OK
       σ *= σ_update
       if norm(state.cx,Inf) > stp.meta.atol
          sub_stp.pb.sigma = σ #Update the FletcherPenaltyNLP
       end
       #reinitialize the State(s)
       #stp.current_state.lambda = nothing only lambda is no longer valid.
       reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
   end
  end #end of main loop

  return GenericExecutionStats(status_stopping_to_stats(stp), stp.pb,
                               solution=stp.current_state.x,
                               objective=stp.current_state.fx,
                               primal_feas=norm(stp.current_state.cx,Inf),
                               dual_feas=norm(stp.current_state.current_score,Inf),
                               multipliers = stp.current_state.lambda,
                               iter=stp.meta.nb_of_stop,
                               elapsed_time=stp.current_state.current_time - stp.meta.start_time)
end
