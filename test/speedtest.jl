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

nlp = ADNLPModel(
  x -> 0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2,
  [2.0; 2.0; 2.0],
  x -> [x[1] + x[3]^2 + 1.0],
  [0.0],
  [0.0],
)

# problem with the 2nd approximation:
@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 30,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> atol, # atol
  rtol_sub = rtol -> rtol, # rtol
  η_1 = 0.0,
  η_update = 1.0,
  subpb_unbounded_threshold = 1 / √eps(Float64),
)
print(nlp.counters)
reset!(nlp)

@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 30,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> 1e-1, # atol
  rtol_sub = rtol -> 1e-1, # rtol
  η_1 = 0.0,
  η_update = 1.0,
  subpb_unbounded_threshold = 1 / √eps(Float64),
)
print(nlp.counters)
reset!(nlp)

@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 30,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> 1e-1, # atol
  rtol_sub = rtol -> 1e-1, # rtol
  η_1 = 1.0,
  η_update = 10.0,
  subpb_unbounded_threshold = 1 / √eps(Float64),
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
[ Info:      7            Optml   4.1e-02   3.4e-04   5.8e-02   3.2e+01          Optimal   0.0e+00
  0.806385 seconds (957.81 k allocations: 57.455 MiB, 3.44% gc time, 90.31% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 101               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 101               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 107   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 17    
           jprod: ███████████████████⋅ 312             jtprod: ████████████████████ 336               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 34    
           hprod: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 56               jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  435.957 ns (3 allocations: 752 bytes)
  7.705 μs (46 allocations: 4.66 KiB)
  8.451 μs (49 allocations: 5.41 KiB)
  18.940 μs (114 allocations: 10.77 KiB)
  23.935 μs (148 allocations: 12.20 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 122103
           jprod: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 381975          jtprod: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 381975            hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 122103
           hprod: ████████████████████ 1423260           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

         =#
#=
LDLt
[ Info:      7            Optml   4.1e-02   3.4e-04   5.8e-02   3.2e+01          Optimal   0.0e+00
  1.200330 seconds (957.81 k allocations: 57.455 MiB, 6.15% gc time, 91.36% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 101               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 101               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 107   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 17    
           jprod: ███████████████████⋅ 312             jtprod: ████████████████████ 336               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 34    
           hprod: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 56               jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  426.503 ns (3 allocations: 752 bytes)
  8.351 μs (46 allocations: 4.66 KiB)
  9.811 μs (49 allocations: 5.41 KiB)
  20.173 μs (113 allocations: 10.72 KiB)
  11.716 μs (81 allocations: 8.98 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 79009 
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 79008 
           hprod: ████████████████████ 1110016           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
=#
