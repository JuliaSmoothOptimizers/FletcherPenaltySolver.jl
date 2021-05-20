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
  max_iter = 40,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> atol, # atol
  rtol_sub = rtol -> rtol, # rtol
  η_1 = 0.,
  η_update = 1.,
)
print(nlp.counters)
reset!(nlp)

@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 40,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> 1e-1, # atol
  rtol_sub = rtol -> 1e-1, # rtol
  η_1 = 0.,
  η_update = 1.,
)
print(nlp.counters)
reset!(nlp)

@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 40,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> 1e-1, # atol
  rtol_sub = rtol -> 1e-1, # rtol
  η_1 = 1.,
  η_update = 10.,
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
 38.886462 seconds (110.48 M allocations: 5.419 GiB, 9.51% gc time, 0.03% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 60    
           jprod: ██████████████████⋅⋅ 651             jtprod: ████████████████████ 731               hess: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  3.030 μs (24 allocations: 1.48 KiB)
  8.459 μs (57 allocations: 4.30 KiB)
  8.736 μs (60 allocations: 5.05 KiB)
  18.771 μs (107 allocations: 8.17 KiB)
  21.828 μs (159 allocations: 11.05 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
           jprod: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 716295          jtprod: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 716295            hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 197580
           hprod: ████████████████████ 2230006           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0    


         =#
#=
LDLt
[ Info:     18            Optml   1.0e-19   1.8e-14   9.7e-09   1.3e+02          Optimal
 39.002745 seconds (110.01 M allocations: 5.392 GiB, 10.09% gc time, 0.03% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 220   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 60    
           jprod: ██████████████████⋅⋅ 651             jtprod: ████████████████████ 731               hess: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 120   
           hprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 160              jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  4.888 μs (34 allocations: 1.64 KiB)
  10.524 μs (67 allocations: 4.45 KiB)
  10.891 μs (70 allocations: 5.20 KiB)
  21.577 μs (116 allocations: 8.28 KiB)
  13.642 μs (102 allocations: 8.73 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 2     
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 167082
           hprod: ████████████████████ 2022774           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0    

=#
