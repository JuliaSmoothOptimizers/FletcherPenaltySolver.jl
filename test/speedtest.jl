using FletcherPenaltyNLPSolver, ADNLPModels, LinearAlgebra, Test, NLPModels
using Random
using BenchmarkTools

Random.seed!(1234)

nlp = ADNLPModel(
  x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, 
  [-1.2; 1.0],
  x -> [sum(x) - 2], 
  [0.0], 
  [0.0],
  name = "Rosenbrock with ∑x = 2",
)

# problem with the 2nd approximation:
@time fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(2), max_iter = 30)
print(nlp.counters)
reset!(nlp)

fpnlp = FletcherPenaltyNLP(nlp)
xr = rand(nlp.meta.nvar)
@btime obj(fpnlp, xr);
@btime grad(fpnlp, xr);
@btime objgrad(fpnlp, xr);
@btime hess(fpnlp, xr);
@btime hprod(fpnlp, xr, xr);

print(nlp.counters)

#=
.
[ Info:     18            Optml   3.1e-17   7.2e-10   2.6e-07   1.3e+02          Optimal
 57.607605 seconds (113.10 M allocations: 5.557 GiB, 7.50% gc time, 0.04% compilation time)
  Counters:
             obj: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               grad: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               cons: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 139   
           jprod: █████████████████⋅⋅⋅ 410             jtprod: ████████████████████ 490               hess: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  14.316 μs (113 allocations: 9.58 KiB)
  9.452 μs (57 allocations: 4.30 KiB)
  9.692 μs (60 allocations: 5.05 KiB)
  19.418 μs (107 allocations: 8.17 KiB)
  22.937 μs (145 allocations: 10.67 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
           jprod: ████████████████⋅⋅⋅⋅ 1100344          jtprod: ████████████████⋅⋅⋅⋅ 1100344            hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 176178
           hprod: ████████████████████ 1448558           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0  
.
=#
#=
[ Info:     18            Optml   1.0e-19   0.0e+00   9.7e-09   1.3e+02          Optimal
 52.011500 seconds (89.74 M allocations: 4.664 GiB, 8.51% gc time, 52.04% compilation time)
  Counters:
             obj: ████████████████⋅⋅⋅⋅ 220               grad: ████████████████⋅⋅⋅⋅ 220               cons: ████████████████⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ████████████████████ 279   
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 80                hess: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: ████████████⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  11.567 μs (112 allocations: 9.55 KiB)
┌ Warning: Failed solving linear system with cg.
└ @ FletcherPenaltyNLPSolver ~/cvs/FletcherPenaltyNLPSolver/src/solve_two_systems.jl:31
  11.130 μs (69 allocations: 4.55 KiB)
  11.359 μs (72 allocations: 5.30 KiB)
  22.277 μs (119 allocations: 8.42 KiB)
  23.582 μs (157 allocations: 10.92 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
           jprod: ████████████████████ 1160122          jtprod: ████████████████████ 1160122            hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 126405
           hprod: ████████████████████ 1149116           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0  

           =#
