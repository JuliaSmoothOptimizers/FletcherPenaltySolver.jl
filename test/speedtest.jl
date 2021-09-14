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
fpnlp = FletcherPenaltyNLP(nlp, qds = qds)
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
[ Info:      3            Optml   4.0e-02   6.9e-09   2.7e-08   4.0e+00          Optimal   0.0e+00
 31.888646 seconds (111.09 M allocations: 5.458 GiB, 9.57% gc time, 0.07% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 114   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 22    
           jprod: ███████████████████⋅ 333             jtprod: ████████████████████ 354               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 44    
           hprod: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 50               jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  340.710 ns (3 allocations: 752 bytes)
  6.863 μs (42 allocations: 4.22 KiB)
  7.126 μs (45 allocations: 4.97 KiB)
  11.411 μs (81 allocations: 8.36 KiB)
  20.745 μs (135 allocations: 11.08 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 361543
           jprod: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 845191          jtprod: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 845191            hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 361543
           hprod: ████████████████████ 2604602           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           =#
#=
LDLt
[ Info:      3            Optml   4.0e-02   6.9e-09   2.7e-08   4.0e+00          Optimal   0.0e+00
 32.433215 seconds (111.56 M allocations: 5.484 GiB, 9.64% gc time, 0.07% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 114   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 22    
           jprod: ███████████████████⋅ 333             jtprod: ████████████████████ 354               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 44    
           hprod: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 50               jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  358.481 ns (3 allocations: 752 bytes)
  6.863 μs (42 allocations: 4.22 KiB)
  7.014 μs (45 allocations: 4.97 KiB)
  11.293 μs (81 allocations: 8.36 KiB)
  12.660 μs (78 allocations: 8.50 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 383683
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 383682
           hprod: ████████████████████ 2842812           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
=#
