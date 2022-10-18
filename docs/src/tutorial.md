# Tutorial

```@contents
Pages = ["tutorial.md"]
```

## FletcherPenaltySolver Tutorial

In this tutorial, we explore on small instances various possibilities offered by `fps_solve` defined in the package `FletcherPenaltySolver`.

### Type stable algorithm

The algorithm is implemented in pure Julia, so if one also chooses an unconstrained solver in pure Julia, we can Julia's type stability to solve optimization problems in a precision different than `Float64`.
In the following example, we use `tron` from [`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) on a simple example in `Float32`.

```@example ex1
using ADNLPModels, FletcherPenaltySolver, JSOSolvers
T = Float32
nlp = ADNLPModel(x -> (1 - x[1])^2, T[-1.2; 1.0], x -> [10 * (x[2] - x[1]^2)], T[0.0], T[0.0])
stats = fps_solve(nlp, hessian_approx = Val(2), subproblem_solver = tron, rtol = T(1e-4), verbose = 1)
(stats.dual_feas, stats.primal_feas, stats.status, typeof(stats.solution))
```

### A factorization-free solver

The main advantage of `fps_solver` is the possibility to use Hessian and Jacobian-vector products only, whenever one uses a subproblem solver with the same property.
So, it is not necessary to compute and store explicitly those matrices.
In the following example, we choose a problem with equality constraints from [`OptimizationProblems.jl`](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl).

```@example ex2
using ADNLPModels, FletcherPenaltySolver, JSOSolvers, OptimizationProblems
nlp = OptimizationProblems.ADNLPProblems.hs28()
stats = fps_solve(nlp, subproblem_solver = tron, qds_solver = :iterative)
(stats.dual_feas, stats.primal_feas, stats.status, stats.elapsed_time)
```
Exploring `nlp`'s counter, we can see that no Hessian or Jacobian matrix has been evaluated.
```@example ex2
nlp.counters
```
We can compare this result with `ipopt` that uses the Jacobian and Hessian matrices.
```@example ex2
using NLPModels, NLPModelsIpopt
reset!(nlp);
stats = fps_solve(nlp, subproblem_solver = ipopt, qds_solver = :iterative)
(stats.dual_feas, stats.primal_feas, stats.status, stats.elapsed_time)
```

```@example ex2
nlp.counters
```

### Stopping-solver

```@example ex3
using ADNLPModels, FletcherPenaltySolver, Stopping
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
c(x) = [x[1]^2 + x[2]^2 - 2]
T = Float64
x0 = T[-1.2; 1.0]
ℓ, u = zeros(T, 2), 2 * ones(T, 2)
nlp = ADNLPModel(f, x0, ℓ, u, c, zeros(1), zeros(1))

stp = NLPStopping(nlp)
stats = fps_solve(stp)
```

It is then possible to explore the various quantities computed by the algorithm.
For instance, recompute the gradient of the Lagrangian.
```@example ex3
state = stp.current_state
state.gx + state.Jx' * state.lambda
```

Another possibility is to reuse the `Stopping` for another solve.
```@example ex3
new_x0 = 4 * ones(2)
reinit!(stp, rstate = true, x = new_x0)
Stopping.reset!(stp.pb)
stats = fps_solve(stp)
```

We refer to [`Stopping.jl`]() and [`https://solverstoppingjulia.github.io`](https://solverstoppingjulia.github.io/StoppingInterface.jl/dev/) for tutorials and documentation.
