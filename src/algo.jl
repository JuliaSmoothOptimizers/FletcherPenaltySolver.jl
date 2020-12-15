function Fletcher_penalty_optimality_check(pb :: AbstractNLPModel, state :: NLPAtX)
    #i) state.cx #<= \epsilon  (1 + \| x k \|_\infty  + \| c(x 0 )\|_\infty  )
    #ii) state.gx <= #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
    #iii) state.res (gradient phi_s) #\epsilon  (1 + \| y k \|  \infty  + \| g \σ  (x 0 )\|  \infty  )
    # returns i) + ii) OR iii) ?
    nxk = max(norm(state.x), 1.)
    nlk = state.lambda == nothing ? 1. : max(norm(state.lambda), 1.)
    
    cx  = state.cx/nxk
    res = state.res/nlk
    
 return vcat(cx, res)
end

struct AlgoData{T <: Real}
    
    #Initialize, Update and Bound parameters of the penalized problem:
    σ_0      :: T
    σ_max    :: T
    σ_update :: T
    ρ_0      :: T
    ρ_max    :: T
    ρ_update :: T
    δ_0      :: T
    
    #Bound on the Lagrange multipliers
    yM       :: T
    
    #Algorithmic parameters
    Δ        :: T #expected decrease in feasibility between two iterations
    
    #Functions used in the algorithm
    linear_system_solver  :: Function
    
    unconstrained_solver  :: Function
end

function AlgoData(T                    :: DataType;
                  σ_0                  :: Real      = one(T),
                  σ_max                :: Real      = eps(T),
                  σ_update             :: Real      = T(1.15),
                  ρ_0                  :: Real      = one(T),
                  ρ_max                :: Real      = eps(T),
                  ρ_update             :: Real      = T(1.15),
                  δ_0                  :: Real      = zero(T),
                  yM                   :: Real      = typemax(T),
                  Δ                    :: Real      = T(0.95),
                  linear_system_solver :: Function  = _solve_with_linear_operator,
                  unconstrained_solver :: Function  = knitro)
                  
   return AlgoData(σ_0, σ_max, σ_update, ρ_0, ρ_max, ρ_update, δ_0, yM, Δ, linear_system_solver, unconstrained_solver)
end

AlgoData(;kwargs...) = AlgoData(Float64;kwargs...)

include("lbfgs.jl")

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
                                 σ_0                   :: Number    = 1.,
                                 σ_max                 :: Number    = 1/eps(),
                                 σ_update              :: Number    = 1.15,
                                 ρ_0                   :: Number    = 1.,
                                 ρ_max                 :: Number    = 1/eps(),
                                 ρ_update              :: Number    = 1.15,
                                 δ_0                   :: Number    = 0.,
                                 linear_system_solver  :: Function  = _solve_with_linear_operator,
                                 unconstrained_solver  :: Function  = knitro,
                                 kwargs...)

 cx0, gx0 = cons(nlp, x0), grad(nlp, x0)
 #Tanj: how to handle stopping criteria where tol_check depends on the State?
 Fptc(atol, rtol, opt0) = rtol * vcat(ones(nlp.meta.ncon) .+ norm(cx0, Inf),
                                      ones(nlp.meta.nvar) .+ norm(gx0, Inf))
                                      
 initial_state = NLPAtX(x0, zeros(nlp.meta.ncon), cx = cx0, gx = gx0, res = gx0)
 stp = NLPStopping(nlp, initial_state,
                   optimality_check = Fletcher_penalty_optimality_check,
                   rtol = 1e-6,
                   tol_check = Fptc,
                   max_cntrs = _init_max_counters(quick = typemax(Int64)); kwargs...)

 return Fletcher_penalty_solver(stp,
                                σ_0 = σ_0, σ_max = σ_max, σ_update = σ_update,
                                ρ_0 = ρ_0, ρ_max = ρ_max, ρ_update = ρ_update,
                                δ_0 = δ_0,
                                linear_system_solver = linear_system_solver,
                                unconstrained_solver = unconstrained_solver)
end

function Fletcher_penalty_solver(stp                   :: NLPStopping;
                                 σ_0                   :: Number    = one(eltype(stp.pb.meta.x0)),
                                 σ_max                 :: Number    = eps(eltype(stp.pb.meta.x0)),
                                 σ_update              :: Number    = eltype(stp.pb.meta.x0)(1.15),
                                 ρ_0                   :: Number    = one(eltype(stp.pb.meta.x0)),
                                 ρ_max                 :: Number    = eps(eltype(stp.pb.meta.x0)),
                                 ρ_update              :: Number    = eltype(stp.pb.meta.x0)(1.15),
                                 δ_0                   :: Real      = zero(eltype(stp.pb.meta.x0)),
                                 linear_system_solver  :: Function  = _solve_with_linear_operator,
                                 unconstrained_solver  :: Function  = knitro)
  state = stp.current_state
  #Initialize parameters
  x0, σ, ρ, δ = state.x, σ_0, ρ_0 , δ_0
  
  #Initialize the unconstrained NLP with Fletcher's penalty function.
  nlp = FletcherPenaltyNLP(stp.pb, σ, ρ, δ, linear_system_solver)

  #First call to the stopping
  OK = start!(stp)
  #Prepare the subproblem-stopping for the unconstrained minimization.
  sub_stp = NLPStopping(nlp, NLPAtX(x0), main_stp = stp, optimality_check = unconstrained_check)
  
  nc0 = norm(state.cx, Inf)
  Δ   = 0.95 #expected decrease in feasibility
  unsuccessful_subpb = 0 #number of consecutive failed subproblem solve.
  stalling = 0 #number of consecutive successful subproblem solve without progress
  restoration_phase = false

  T = eltype(x0)
  
  @info log_header([:iter, :f, :c, :score, :σ, :stat], [Int, T, T, T, T, Symbol],
                   hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :σ=>"σ"))
  @info log_row(Any[0, NaN, nc0, norm(state.current_score,Inf), σ, :o])

  while !OK #main loop

   #Solve the subproblem
   reinit!(sub_stp) #reinit the sub-stopping.
   sub_stp = unconstrained_solver(sub_stp)
   #Update the State with the info given by the subproblem:
   if sub_stp.meta.optimal
      if sub_stp.current_state.x == state.x
       stalling += 1  
      end
      unsuccessful_subpb = 0
      
      Stopping.update!(state, x      = sub_stp.current_state.x,
                              fx     = sub_stp.pb.fx,
                              gx     = sub_stp.pb.gx,
                              cx     = sub_stp.pb.cx,
                              lambda = sub_stp.pb.ys,
                              res    = sub_stp.current_state.gx)
      
   elseif sub_stp.meta.unbounded
       #Penalized problem is unbounded...
       unsuccessful_subpb += 1
   elseif sub_stp.meta.iteration_limit || sub_stp.meta.tired || sub_stp.meta.resources || sub_stp.meta.stalled
       #How to control these parameters in knitro ??
       unsuccessful_subpb += 1
   else
       stp.meta.fail_sub_pb = true
   end
   
   #Check optimality conditions: either stop! is true OR the penalty parameter is too small
   stp.meta.fail_sub_pb = σ > σ_max || ρ > ρ_max #stp.meta.stalled 
   OK = stop!(stp)

   @info log_row(Any[stp.meta.nb_of_stop, 
                     state.fx, norm(state.cx), norm(state.res), 
                     norm(state.gx), σ, status(sub_stp)])
 
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
          σ /= ρ_update^3
          sub_stp.pb.σ = σ
          sub_stp.pb.ρ /= ρ_update^3
          #reinitialize the State(s) as the problem changed
          reinit!(sub_stp.current_state, x = state.x) #reinitialize the State (keeping x)
             
          stalling = 0
          unsuccessful_subpb = 0
       elseif ncx > Δ * nc0 && !feas
          σ *= σ_update
          sub_stp.pb.σ = σ
          sub_stp.pb.ρ *= ρ_update
          #reinitialize the State(s) as the problem changed
          reinit!(sub_stp.current_state) #reinitialize the State (keeping x)
       end
       nc0 = copy(ncx)
   end
   
  end #end of main loop
  
  @show status(stp), restoration_phase
  return GenericExecutionStats(status_stopping_to_stats(stp), stp.pb,
                               solution     = stp.current_state.x,
                               objective    = stp.current_state.fx,
                               primal_feas  = norm(stp.current_state.cx, Inf),
                               dual_feas    = norm(stp.current_state.res, Inf),
                               multipliers  = stp.current_state.lambda,
                               iter         = stp.meta.nb_of_stop,
                               elapsed_time = stp.current_state.current_time - stp.meta.start_time)
end
