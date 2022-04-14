using Pkg; Pkg.activate("")
using CUTEst, NLPModels, NLPModelsIpopt, DCISolver, SolverBenchmark
using FletcherPenaltyNLPSolver

problems = readlines("list_problems.dat")
cutest_problems = (CUTEstModel(p) for p in problems)

max_time = 1200.0 #20 minutes
tol = 1e-5

solvers = Dict(
  :ipopt => nlp -> ipopt(
      nlp,
      print_level = 0,
      dual_inf_tol = Inf,
      constr_viol_tol = Inf,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      max_cpu_time = max_time,
      tol = tol,
  ),
  :dcildl => nlp -> dci(
      nlp,
      linear_solver = :ldlfact,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = tol,
      ctol = tol,
      rtol = tol,
  ),
  :fpsff =>
    nlp -> fps_solve(
      nlp,
      nlp.meta.x0,
      atol = tol,
      rtol = tol,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      qds_solver = :iterative,
      unconstrained_solver = ipopt,
      ls_atol = tol, # √eps(),
      ls_rtol = tol, # √eps(),
      # ls_itmax = 5 * (nlp.meta.ncon + nlp.meta.nvar),
      ln_atol = tol, # √eps(),
      ln_rtol = tol, # √eps(),
      # ln_btol = √eps(),
      # ln_conlim = 1 / √eps(),
      # ln_itmax = 5 * (nlp.meta.ncon + nlp.meta.nvar),
      ne_atol = tol, # √eps()/100,
      ne_rtol = tol, # √eps()/100,
      # ne_ratol = zero(Float64),
      # ne_rrtol = zero(Float64),
      # ne_etol = √eps(),
      # ne_itmax = 0,
      # ne_conlim = 1 / √eps(),
      #        atol_sub = atol -> 1e-1, # atol,
      #        rtol_sub = rtol -> 1e-1, # rtol,
      #        η_1 = 1.,
      #        η_update = 10.,
    ),
  :fps =>
    nlp -> fps_solve(
      nlp,
      nlp.meta.x0,
      atol = tol,
      rtol = tol,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      qds_solver = :ldlt,
      ldlt_tol = √eps(),
      ldlt_r1 = √eps(),
      ldlt_r2 = -√eps(),
      unconstrained_solver = ipopt,
      #        atol_sub = atol -> 1e-1, # atol,
      #        rtol_sub = rtol -> 1e-1, # rtol,
      #        η_1 = 1.,
      #        η_update = 10.,
    ),
)
stats = bmark_solvers(solvers, cutest_problems)

using JLD2
@save "ipopt_dcildl_fps_$(string(length(problems))).jld2" stats
