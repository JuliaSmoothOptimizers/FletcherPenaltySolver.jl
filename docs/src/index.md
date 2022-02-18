# FletcherPenaltyNLPSolver - Fletcher's Penalty Method

[![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] [![build-ci][build-ci-img]][build-ci-url] [![codecov][codecov-img]][codecov-url] [![release][release-img]][release-url] [![doi][doi-img]][doi-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://tmigot.github.io/FletcherPenaltyNLPSolver/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://tmigot.github.io/FletcherPenaltyNLPSolver/dev
[build-ci-img]: https://github.com/tmigot/FletcherPenaltyNLPSolver/workflows/CI/badge.svg?branch=main
[build-ci-url]: https://github.com/tmigot/FletcherPenaltyNLPSolver/actions
[codecov-img]: https://codecov.io/gh/tmigot/FletcherPenaltyNLPSolver/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/tmigot/FletcherPenaltyNLPSolver
[release-img]: https://img.shields.io/github/v/release/tmigot/FletcherPenaltyNLPSolver.svg?style=flat-square
[release-url]: https://github.com/tmigot/FletcherPenaltyNLPSolver/releases
[doi-img]: https://joss.theoj.org/papers/10.21105/joss.03991/status.svg
[doi-url]: https://doi.org/10.21105/joss.03991

FPS is a solver for equality-constrained nonlinear problems, i.e.,
optimization problems of the form

    min f(x)     s.t.     c(x) = 0,  ℓ ≤ x ≤ u.

It uses other JuliaSmoothOptimizers packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
If a general inequality-constrained problem is given to the solver, it solves the problem reformulated as a `SlackModel` from [NLPModelsModifiers.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl).


We refer to [juliasmoothoptimizers.github.io](https://juliasmoothoptimizers.github.io) for tutorials on the NLPModel API. This framework allows the usage of models from Ampl (using [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)), CUTEst (using [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)), JuMP (using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)), PDE-constrained optimization problems (using [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)) and models defined with automatic differentiation (using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl)).

## Algorithm

The function `fps_solver` solves a nonlinear optimization problem by iteratively solving 
the bound-constrained optimization problem using Fletcher penalty function:

```math
         \begin{aligned}
         \min_{x} \ & f(x) - c(x)^T \lambda_\delta(x) + \frac{\rho}{2}\|c(x)\|^2_2, \\
         \mbox{where } \lambda_\delta(x) \in \arg\min_{y} \frac{1}{2}\| \nabla c(x)^T y - \nabla f(x) \|^2_2 + \sigma c(x)^T y + \frac{\delta}{2}\|y\|^2.
         \end{aligned}
```

For equality-constrained problems, the method iteratively solves an unconstrained problem. For bound and equality-constrained problems, the subproblems are bound-constrained problems. Any solver compatible with [Stopping](https://github.com/vepiteski/Stopping.jl) can be used.
By default, we use `ipopt` from [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) to solve the subproblem, but other solvers can be used such as `knitro` from [NLPModelsKnitro.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl) or any solvers from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl). The Stopping version of these solvers is available in [StoppingInterface](https://github.com/SolverStoppingJulia/StoppingInterface.jl).

It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to evaluate the derivatives of the penalized subproblem, but one can also use a matrix-free version with [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

## References

> Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
> Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
> SIAM Journal on Scientific Computing, 42(3), A1809-A1835.
> [10.1137/19M1238265](https://doi.org/10.1137/19M1238265)

## How to Cite

If you use FletcherPenaltyNLPSolver in your work, please cite using the format given in [CITATION.bib](https://github.com/tmigot/FletcherPenaltyNLPSolver/blob/main/CITATION.bib).

## Installation

1. `pkg> add https://github.com/tmigot/FletcherPenaltyNLPSolver`

## Example

We consider in this example the minization of the Rosenbrock function.
```@example
    \min_x \ 100 * (x₂ - x₁²)² + (x₁ - 1)²,
```
The problem is modeled using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) with `[-1.2; 1.0]` as default initial point, and then solved using `fps_solve`.

```julia
using FletcherPenaltyNLPSolver, ADNLPModels

# Rosenbrock
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = fps_solve(nlp)
```

We consider in this example the minization of the Rosenbrock function over an inequality constraint.
```math
    \min_x \ 100 * (x₂ - x₁²)² + (x₁ - 1)² \quad \text{s.t.} \quad  0 ≤ x₁x₂ ≤ 1,
```
The problem is modeled using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) with `[-1.2; 1.0]` as default initial point, and then solved using `fps_solve`.

```@example
using FletcherPenaltyNLPSolver, ADNLPModels

nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
  [-1.2; 1.0],
  x->[x[1] * x[2]],
  [0.0],
  [1.0],
)
stats = fps_solve(nlp)
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/FletcherPenaltyNLPSolver/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
