# FletcherPenaltyNLPSolver

The algorithm depends on Stopping (version > 0.2.3)
```julia
pkg> add https://github.com/vepiteski/Stopping.jl
pkg> test Stopping
pkg> status Stopping
```

##TODO

Solve the unconstrained problem:
- Add *hess_structure!* (*hess_coord!*)  to use [NLPModelsKnitro](https://github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl) (obj, cons!, grad!, hprod!)
- Same comment to apply [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl).
