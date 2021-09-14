# FletcherPenaltyNLPSolver

![CI](https://github.com/tmigot/FletcherPenaltyNLPSolver/workflows/CI/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/tmigot/FletcherPenaltyNLPSolver/branch/main/graph/badge.svg)](https://codecov.io/gh/tmigot/FletcherPenaltyNLPSolver)

This implementation uses [Stopping](https://github.com/vepiteski/Stopping.jl).

## Algorithm

The function `Fletcher_penalty_solver(nlp :: AbstractNLPModel)` solves a nonlinear
optimization problem by iteratively solving 
the bound-constrained optimization problem using Fletcher penalty function:

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
