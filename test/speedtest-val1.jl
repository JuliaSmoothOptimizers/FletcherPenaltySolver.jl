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

#=
n = 300
  nlp = ADNLPModel(x -> dot(x, x), zeros(n), x -> [sum(x) - 1], zeros(1), zeros(1))
=#

# problem with the 2nd approximation:
@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(1),
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
#=
@time fps_solve(
  nlp,
  nlp.meta.x0,
  hessian_approx = Val(2),
  max_iter = 30,
  qds_solver = :iterative, # :iterative or :ldlt
  atol_sub = atol -> 1e-3, # atol
  rtol_sub = rtol -> 1e-3, # rtol
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
=#
qds = FletcherPenaltyNLPSolver.IterativeSolver(nlp, 0.0)
# qds = FletcherPenaltyNLPSolver.LDLtSolver(nlp, 0.0)
fpnlp = FletcherPenaltyNLP(nlp, qds = qds, hessian_approx = Val(1))
xr = rand(nlp.meta.nvar)
@btime obj(fpnlp, xr);
gx = similar(xr)
@btime grad!(fpnlp, xr, gx);
@btime objgrad!(fpnlp, xr, gx);
@btime hess_coord(fpnlp, xr);
@btime hprod!(fpnlp, xr, xr, gx);

print(nlp.counters)

rhs1 = grad(fpnlp, xr);
rhs2 = rand(nlp.meta.nvar);
@btime FletcherPenaltyNLPSolver.solve_two_least_squares(fpnlp, xr, rhs1, rhs2)
# 12.987 μs (91 allocations: 6.14 KiB)

rhs2 = cons(fpnlp.nlp, xr)
@btime FletcherPenaltyNLPSolver.solve_two_mixed(fpnlp, xr, rhs1, rhs2);
#  19.561 μs (76 allocations: 4.86 KiB)

#=
Iterative
[ Info:      3            Optml   4.0e-02   6.3e-09   2.6e-08   4.0e+00          Optimal   0.0e+00
  80.473073 seconds (116.98 M allocations: 5.798 GiB, 7.36% gc time, 0.06% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 74                grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 74                cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 80    
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16    
           jprod: ███████████████████⋅ 231             jtprod: ████████████████████ 245               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 32    
           hprod: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 36               jhess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16              jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  453.888 ns (3 allocations: 752 bytes)
  11.843 μs (42 allocations: 4.22 KiB)
  8.695 μs (45 allocations: 4.97 KiB)
  28.684 μs (143 allocations: 14.69 KiB)
  44.942 μs (253 allocations: 20.31 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 58830 
           jprod: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 307527          jtprod: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 307527            hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 58830 
           hprod: ████████████████████ 750118           jhess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 58830           jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  23.608 μs (91 allocations: 6.14 KiB)
  10.482 μs (76 allocations: 4.86 KiB)
       
           =#
#=
LDLt
[ Info:      3            Optml   4.0e-02   6.3e-09   2.6e-08   4.0e+00          Optimal   0.0e+00
  9.716252 seconds (6.88 M allocations: 397.899 MiB, 4.25% gc time, 83.01% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 74                grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 74                cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 80    
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16    
           jprod: ███████████████████⋅ 231             jtprod: ████████████████████ 245               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 32    
           hprod: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 36               jhess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16              jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  439.389 ns (3 allocations: 752 bytes)
  9.328 μs (42 allocations: 4.22 KiB)
  9.090 μs (45 allocations: 4.97 KiB)
  30.408 μs (143 allocations: 14.69 KiB)
  39.356 μs (197 allocations: 17.80 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 54853 
           jprod: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 159219          jtprod: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 159219            hess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 54852 
           hprod: ████████████████████ 818082           jhess: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 54852           jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  1.608 μs (27 allocations: 3.27 KiB)
  11.167 μs (64 allocations: 5.77 KiB)

           =#
