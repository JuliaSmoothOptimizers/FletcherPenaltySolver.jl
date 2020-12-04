using CUTEst

function runcutest()
  problems = filter(x->length(x) <= 5, CUTEst.select(only_free_var=true, only_equ_con=true))
  sort!(problems)
  @printf("%-7s  %15s  %15s  %15s\n",
          "Problem", "f(x)", "‖∇ℓ(x,λ)‖", "‖c(x)‖")
  for p in problems
    nlp = CUTEstModel(p)
    try
      x, fx, nlx, ncx = Fletcher_penalty_solver(nlp)
      @printf("%-7s  %15.8e  %15.8e  %15.8e\n", p, fx, nlx, ncx)
    catch
      @printf("%-7s  %s\n", p, "failure")
    finally
      finalize(nlp)
    end
  end
end

nlp = CUTEstModel("BT1")
Fletcher_penalty_solver(nlp, nlp.meta.x0)
finalize(nlp)
