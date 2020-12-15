# FletcherPenaltyNLPSolver

The algorithm depends on Stopping (version > 0.2.4)
```julia
pkg> add https://github.com/vepiteski/Stopping.jl
pkg> test Stopping
pkg> status Stopping
```

## Algorithm

The function `Fletcher_penalty_solver(nlp :: AbstractNLPModel)` solves a nonlinear
optimization problem with **equality constraints** by iteratively solving 
the unconstrained optimization problem using Fletcher penalty function:

```math
         \begin{aligned}
         \min_{x} \ & f(x) - c(x)^T \lambda_\delta(x) + \frac{\rho}{2}\|c(x)\|^2_2, \\
         \mbox{where } \lambda_\delta(x) \in \arg\min_{y} \frac{1}{2}\| \nabla c(x)^T y - \nabla f(x) \|^2_2 + \sigma c(x)^T y + \frac{\delta}{2}\|y\|^2.
         \end{aligned}
```


## References

Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
  Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
  SIAM Journal on Scientific Computing, 42(3), A1809-A1835.
