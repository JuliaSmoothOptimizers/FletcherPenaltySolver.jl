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
@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 30,
  qds_solver = :iterative, # :iterative or :ldlt
)
print(nlp.counters)
reset!(nlp)

qds = FletcherPenaltyNLPSolver.IterativeSolver(nlp, 0.0)
# qds = FletcherPenaltyNLPSolver.LDLtSolver(nlp, 0.0)
fpnlp = FletcherPenaltyNLP(
  nlp,
  qds = qds,
)
xr = rand(nlp.meta.nvar)
@btime obj(fpnlp, xr);
@btime grad(fpnlp, xr);
@btime objgrad(fpnlp, xr);
@btime hess(fpnlp, xr);
@btime hprod(fpnlp, xr, xr);

print(nlp.counters)

#=
Iterative
[ Info:     18            Optml   1.0e-19   1.8e-14   9.7e-09   1.3e+02          Optimal
 62.950599 seconds (107.74 M allocations: 5.260 GiB, 8.21% gc time, 0.03% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 60    
           jprod: ██████████████████⋅⋅ 651             jtprod: ████████████████████ 731               hess: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  2.979 μs (24 allocations: 1.48 KiB)
  8.535 μs (57 allocations: 4.30 KiB)
  8.908 μs (60 allocations: 5.05 KiB)
  20.089 μs (107 allocations: 8.17 KiB)
  21.254 μs (145 allocations: 10.67 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
           jprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 458623          jtprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 458623            hess: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 82773 
           hprod: ████████████████████ 2014100           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     .
=#
#=
LDLt
[ Info:     18            Optml   1.0e-19   1.8e-14   9.7e-09   1.3e+02          Optimal
  3.126029 seconds (6.21 M allocations: 357.829 MiB, 8.03% gc time, 95.00% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 60    
           jprod: ██████████████████⋅⋅ 651             jtprod: ████████████████████ 731               hess: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  5.224 μs (34 allocations: 1.64 KiB)
  10.943 μs (67 allocations: 4.45 KiB)
  11.390 μs (70 allocations: 5.20 KiB)
  25.278 μs (117 allocations: 8.33 KiB)
  32.366 μs (189 allocations: 15.37 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 79755 
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 121158
           hprod: ████████████████████ 1020282           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           =#
