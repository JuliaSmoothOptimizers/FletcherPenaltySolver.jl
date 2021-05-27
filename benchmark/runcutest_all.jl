using Pkg;
Pkg.activate("bench");
using CUTEst, NLPModels, NLPModelsKnitro, NLPModelsIpopt, SolverBenchmark, SolverTools
#This package
using FletcherPenaltyNLPSolver
#
using Dates, JLD2
using Random

Random.seed!(1234)

function runcutest(; today::String = string(today()))

  #pnames = readlines("paper-problems.list")
  _pnames = CUTEst.select(
    max_var = 300,
    min_con = 1,
    max_con = 300,
    only_free_var = true,
    objtype = 3:6,
  )

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x -> occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  #Same time limit for all the solvers
  max_time = 180.0

  tol_der = 1e-5

  solvers = Dict(
    :ipopt =>
      nlp -> ipopt(
        nlp,
        print_level = 0,
        dual_inf_tol = Inf,
        constr_viol_tol = Inf,
        compl_inf_tol = Inf,
        acceptable_iter = 0,
        max_cpu_time = max_time,
        x0 = nlp.meta.x0,
      ),
    :knitro =>
      nlp -> knitro(
        nlp,
        out_hints = 0,
        outlev = 0,
        feastol = 1e-5,
        feastol_abs = 1e-5,
        opttol = 1e-5,
        opttol_abs = 1e-5,
        maxtime_cpu = max_time,
        x0 = nlp.meta.x0,
      ),
    :FPS =>
      nlp -> fps_solve(
        nlp,
        nlp.meta.x0,
        atol = 1e-5,
        rtol = 1e-5,
        max_time = max_time,
        max_iter = typemax(Int64),
        max_eval = typemax(Int64),
        qds_solver = :ldlt,
        ldlt_tol = √eps(),
        ldlt_r1 = √eps(),
        ldlt_r2 = -√eps(),
#        atol_sub = atol -> 1e-1, # atol,
#        rtol_sub = rtol -> 1e-1, # rtol,
#        η_1 = 1.,
#        η_update = 10.,
      ),
    :FPSFF =>
      nlp -> fps_solve(
        nlp,
        nlp.meta.x0,
        atol = 1e-5,
        rtol = 1e-5,
        max_time = max_time,
        max_iter = typemax(Int64),
        max_eval = typemax(Int64),
        qds_solver = :iterative,
        ls_atol = tol_der, # √eps(),
        ls_rtol = tol_der, # √eps(),
        # ls_itmax = 5 * (nlp.meta.ncon + nlp.meta.nvar),
        ln_atol = tol_der, # √eps(),
        ln_rtol = tol_der, # √eps(),
        # ln_btol = √eps(),
        # ln_conlim = 1 / √eps(),
        # ln_itmax = 5 * (nlp.meta.ncon + nlp.meta.nvar),
        ne_atol = tol_der, # √eps()/100,
        ne_rtol = tol_der, # √eps()/100,
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
  )

  list = ""
  for solver in keys(solvers)
    list = string(list, "_$(solver)")
  end

  stats = bmark_solvers(solvers, cutest_problems)

  @save "$(today)_all_$(list)_$(string(length(pnames))).jld2" stats

  return stats
end

stats = runcutest()
