using CUTEst, NLPModels, NLPModelsIpopt, Plots, SolverBenchmark, SolverTools
#This package
using FletcherPenaltyNLPSolver
gr()

function runcutest()
  _pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true, objtype=3:6)
  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  solvers = Dict(:FPS => nlp -> Fletcher_penalty_solver(nlp, nlp.meta.x0), 
                 :ipopt => (nlp; kwargs...) -> ipopt(nlp, print_level=0, kwargs...))
  stats = bmark_solvers(solvers, cutest_problems)

  #join_df = join(stats, [:objective, :dual_feas, :primal_feas, :neval_obj, :status], invariant_cols=[:name])
  #SolverBenchmark.markdown_table(stdout, join_df)
  for col in [:neval_obj, :neval_cons, :neval_grad, :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :elapsed_time]
    empty = false
    for df in values(stats)
      if all(df[col] .== 0)
        empty = true
      end
    end

    if !empty
    ϵ = minimum(minimum(filter(x -> x > 0, df[col])) for df in values(stats))
    cost(df) = (max.(df[col], ϵ) + (df.status .!= :first_order) * Inf)
    performance_profile(stats, cost)
    png("perf-$col")
    end
  end
end

runcutest()
