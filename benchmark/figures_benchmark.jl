using Pkg;
Pkg.activate("bench");
using Dates, JLD2, SolverTools, SolverBenchmark
using Plots;
pgfplots();

function figure()
  names = ["2021-03-03__FPS_knitro_ipopt_45"]

  tod = string(today())
  dsolvers = [:ipopt, :knitro, :DCILDL, :DCIMA57]
  list = ""
  for solver in dsolvers
    list = string(list, "_$(solver)")
  end
  legend = Dict(
    :neval_obj => "number of f evals",
    :neval_cons => "number of c evals",
    :neval_grad => "number of ∇f evals",
    :neval_jac => "number of ∇c evals",
    :neval_jprod => "number of ∇c*v evals",
    :neval_jtprod => "number of ∇cᵀ*v evals",
    :neval_hess => "number of ∇²f evals",
    :elapsed_time => "elapsed time",
  )
  styles = [:solid, :dash, :dot, :dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
  perf_title(col) = "Performance profile on CUTEst w.r.t. $(string(legend[col]))"

  @load string(names[1], ".jld2") stats

  for col in keys(legend)
    empty = false
    for df in values(stats)
      if all(df[col] .== 0)
        empty = true
      end
    end

    if !empty
      ϵ = minimum(minimum(filter(x -> x > 0, df[col])) for df in values(stats))
      first_order(df) = df.status .== :first_order
      unbounded(df) = df.status .== :unbounded
      solved(df) = first_order(df) .| unbounded(df)
      cost(df) = (max.(df[col], ϵ) + .!solved(df) .* Inf)
      #p = performance_profile(stats, cost)
      p = performance_profile(
        stats,
        cost,
        title = perf_title(col),
        legend = :bottomright,
        linestyles = styles,
      )
      savefig("$(tod)_$(list)_perf-$col.tex")
      png("$(tod)_$(list)_perf-$col")
      #profile_solvers(stats, [cost], ["$(col)"])
      costs = [cost]
      solvers = collect(keys(stats))
      nsolvers = length(solvers)
      npairs = div(nsolvers * (nsolvers - 1), 2)
      colors = get_color_palette(:auto, nsolvers)
      if nsolvers > 2
        ipairs = 0
        # combinations of solvers 2 by 2
        for i = 2:nsolvers
          for j = 1:(i - 1)
            ipairs += 1
            pair = [solvers[i], solvers[j]]
            dfs = (stats[solver] for solver in pair)
            Ps = [hcat([cost(df) for df in dfs]...) for cost in costs]

            clrs = [colors[i], colors[j]]
            stls = [styles[i], styles[j]]
            p = performance_profile(
              Ps[1],
              string.(pair),
              palette = clrs,
              legend = :bottomright,
              styles = stls,
            )
            ipairs < npairs && xlabel!(p, "")
            savefig("$(tod)_$(solvers[i])_$(solvers[j])_perf-$col.tex")
            png("$(tod)_$(solvers[i])_$(solvers[j])_perf-$col")
          end
        end
      else
      end
    end
  end
end

figure()
