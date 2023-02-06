# FletcherPenaltySolver.jl - Fletcher's Penalty Method

[![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] [![build-ci][build-ci-img]][build-ci-url] [![codecov][codecov-img]][codecov-url] [![release][release-img]][release-url] [![doi][doi-img]][doi-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/FletcherPenaltySolver.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/FletcherPenaltySolver.jl/dev
[build-ci-img]: https://github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/workflows/CI/badge.svg?branch=main
[build-ci-url]: https://github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/actions
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSmoothOptimizers/FletcherPenaltySolver.jl
[release-img]: https://img.shields.io/github/v/release/JuliaSmoothOptimizers/FletcherPenaltySolver.jl.svg?style=flat-square
[release-url]: https://github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/releases
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.7153564.svg
[doi-url]: https://doi.org/10.5281/zenodo.7153564

FPS is a solver for equality-constrained nonlinear problems, i.e.,
optimization problems of the form

    min f(x)     s.t.     c(x) = 0.

It uses other [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io/) packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
If a general inequality-constrained problem is given to the solver, it solves the problem reformulated as a `SlackModel` from [NLPModelsModifiers.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl).

## Algorithm

For equality-constrained problems, the method iteratively solves an unconstrained problem. For bound and equality-constrained problems, the subproblems are bound-constrained problems. Any solver compatible with [Stopping.jl](https://github.com/SolverStoppingJulia/Stopping.jl) can be used.
By default, we use `ipopt` from [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) to solve the subproblem, but other solvers can be used such as `knitro` from [NLPModelsKnitro.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl) or any solvers from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl). The Stopping version of these solvers is available in [StoppingInterface.jl](https://github.com/SolverStoppingJulia/StoppingInterface.jl).

It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to evaluate the derivatives of the penalized subproblem, but one can also use a matrix-free version with [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

## References

> Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
> Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
> SIAM Journal on Scientific Computing, 42(3), A1809-A1835.
> [10.1137/19M1238265](https://doi.org/10.1137/19M1238265)

## How to Cite

If you use FletcherPenaltySolver.jl in your work, please cite using the format given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/blob/main/CITATION.cff).

## Installation

`pkg> add FletcherPenaltySolver`

## Example

```julia
using FletcherPenaltySolver, ADNLPModels

# Rosenbrock
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = fps_solve(nlp)

# Constrained
nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
  [-1.2; 1.0],
  x->[x[1] * x[2] - 1],
  [0.0],[1.0],
)
stats = fps_solve(nlp)
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
