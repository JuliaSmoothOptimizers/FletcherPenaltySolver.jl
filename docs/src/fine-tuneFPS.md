# Advanced usage of FletcherPenaltyNLPSolver

## Contents

```@contents
Pages = ["fine-tuneFPS.md"]
```

The main function exported by this package is the function `fps_solve` whose basic usage has been illustrated previously.
It is also possible to fine-tune the parameters used in the implementation in two different ways.

## Examples

FletcherPenaltyNLPSolver exports the function `fps_solve`:
```
   fps_solve(nlp::AbstractNLPModel, x0::AbstractVector{T} = nlp.meta.x0; subsolver_verbose::Int = 0, lagrange_bound = 1 / sqrt(eps(T)), kwargs...)
   fps_solve(stp::NLPStopping; subsolver_verbose::Int = 0, lagrange_bound = 1 / sqrt(eps()), kwargs...)
   fps_solve(stp::NLPStopping, fpssolver::FPSSSolver{T, QDS, US}; subsolver_verbose::Int = 0, lagrange_bound::T = 1 / sqrt(eps(T)))
```
It is, therefore, possible to either call `fps_solve(nlp, x, kwargs...)` and the keywords arguments are passed to both `NLPStopping` and/or `FPSSSolver` constructor or build an instance of `NLPStopping` and/or `FPSSSolver` directly.

```@example ex1
using ADNLPModels, FletcherPenaltyNLPSolver

nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, 
  [-1.2; 1.0],
  x->[x[1] * x[2] - 1], 
  [0.0], [0.0],
  name = "Rosenbrock with x₁x₂=1"
)
stats = fps_solve(nlp)
```

The alternative using `NLPStopping`, see `Stopping.jl`, allow to reuse the same memory if one would re-solve a problem of the same dimension

```@example ex1
using ADNLPModels, FletcherPenaltyNLPSolver, Stopping
stp = NLPStopping(nlp)
stats = fps_solve(stp)
stp.current_state.x .= rand(2)
stats = fps_solve(stp)
```

The `FPSSSolver`, see [`FPSSSolver`](@ref), contains all the metadata and additional pre-allocated memory used by the algorithm.

```@example ex1
stp = NLPStopping(nlp)
data = FPSSSolver(nlp, Float64; kwargs...)
stats = fps_solve(stp, data)
```

## List of possible options

Find below a list of the main options of `fps_solve`.

### Tolerances on the problem

We use `Stopping.jl` to control the algorithmic flow, we refer to [`Stopping.jl`]() and [`https://solverstoppingjulia.github.io`](https://solverstoppingjulia.github.io/StoppingInterface.jl/dev/) for tutorials and documentation.
By default, we use the function `Fletcher_penalty_optimality_check` as optimality check, and the default `tol_check` is `rtol [1 + c(x₀); 1 + ∇f(x₀)]` with `rtol = 1e-7`.

Additional parameters used in stopping the algorithm are defined in the following table.

| Parameters           | Type          | Default      | Description                                    |
| -------------------- | ------------- | ------------ | ---------------------------------------------- |
| lagrange_bound | Real| 1 / sqrt(eps(T)) | bounds on estimated Lagrange multipliers. |
| subsolver_max_iter | Real | 20000 | maximum iteration for the subproblem solver. |

### Algorithmic parameters

The metadata is defined in a `AlgoData` structure at the initialization of `FPSSolver`.

| Parameters    | Type          | Default  | Description                                                                                               |
| ------------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| σ_0 | Real | 1e3 | Initialize subproblem's parameter σ |
| σ_max | Real | 1 / √eps(T) | Maximum value for subproblem's parameter σ | 
| σ_update | Real | T(2) | Update subproblem's parameter σ | 
| ρ_0 | Real | T(2) | Initialize subproblem's parameter ρ | 
| ρ_max | Real | 1 / √eps(T) | Maximum value for subproblem's parameter ρ | 
| ρ_update | Real | T(2) | Update subproblem's parameter ρ | 
| δ_0 | Real | √eps(T) | Initialize subproblem's parameter δ | 
| δ_max | Real | 1 / √eps(T) | Maximum value for subproblem's parameter δ | 
| δ_update | Real | T(10) | Update subproblem's parameter δ | 
| η_1 | Real | zero(T) | Initialize subproblem's parameter η | 
| η_update | Real | one(T) | Update subproblem's parameter η | 
| yM | Real | typemax(T) |  bound on the Lagrange multipliers | 
| Δ | Real | T(0.95) | expected decrease in feasibility between two iterations | 
| subproblem_solver | Function | StoppingInterface.is_knitro_installed ? NLPModelsKnitro.knitro : ipopt | solver used for the subproblem, see also [`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) | 
| subpb_unbounded_threshold | Real | 1 / √eps(T) | below the opposite of this value, the subproblem is unbounded | 
| atol_sub | Function | atol -> atol | absolute tolerance for the subproblem in function of `atol` | 
| rtol_sub | Function | rtol -> rtol | relative tolerance for the subproblem in function of `rtol` | 
| hessian_approx | either `Val(1)` or `Val(2)` | Val(2) | it selects the hessian approximation | 
| convex_subproblem | Bool | false |  true if the subproblem is convex. Useful to set the `convex` option in `knitro`. | 
| qds_solver | Symbol | :ldlt | Initialize the `QDSolver` to solve quasi-definite systems, either `:ldlt` or `:iterative`. |

### Feasibility step

The metadata for the feasibility procedure is defined in a `GNSolver` structure at the initialization of `FPSSolver`.

| Parameters             | Type                                    | Default                                            | Description                                                                                               |
| ---------------------- | --------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| η₁                | Real                          | 1e-3                                               | Feasibility step: decrease the trust-region radius when Ared/Pred < η₁.                                   |
| η₂                | Real                          | 0.66                                               | Feasibility step: increase the trust-region radius when Ared/Pred > η₂.                                   |
| σ₁                | Real                          | 0.25                                               | Feasibility step: decrease coefficient of the trust-region radius.                                        |
| σ₂                | Real                          | 2.0                                                | Feasibility step: increase coefficient of the trust-region radius.                                        |
| Δ₀                | Real                          | 1.0                                                | Feasibility step: initial radius.                                                                         |
| feas_expected_decrease | Real                          | 0.95                                               | Feasibility step: bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease.                             |
| bad_steps_lim          | Integer                                 | 3                                                  | Feasibility step: consecutive bad steps before using a second order step.                                 |
| TR_compute_step        | KrylovSolver                                  | LsmrSolver                                           | Compute the direction in feasibility step.                                           |
| aggressive_step | KrylovSolver | CgSolver | Compute the (aggressive) direction in feasibility step. |
