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
    only_equ_con = true,
    objtype = 3:6,
  )

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x -> occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  #Same time limit for all the solvers
  max_time = 180.0

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
        atol_sub = atol -> atol,
        rtol_sub = rtol -> rtol,
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
        linear_system_solver = FletcherPenaltyNLPSolver._solve_with_linear_operator,
        atol_sub = atol -> atol,
        rtol_sub = rtol -> rtol,
      ),
  )

  list = ""
  for solver in keys(solvers)
    list = string(list, "_$(solver)")
  end

  stats = bmark_solvers(solvers, cutest_problems)

  @save "$(today)_bis_$(list)_$(string(length(pnames))).jld2" stats

  return stats
end

stats = runcutest()
