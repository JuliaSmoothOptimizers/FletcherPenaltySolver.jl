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

include("problems/controlinvestment.jl")

nlp = controlinvestment_autodiff(n=1000)

# problem with the 2nd approximation:
#=
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
=#
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
fpnlp = FletcherPenaltyNLP(
  nlp,
  qds = qds,
)
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

@show "Compare jacobian formation"
using SparseArrays
x = rand(fpnlp.meta.nvar);
@btime fpnlp.Ax = jac(fpnlp.nlp, x);
@btime fpnlp.Ax = sparse(fpnlp.Arows, fpnlp.Acols, jac_coord!(fpnlp.nlp, x, fpnlp.Avals), fpnlp.nlp.meta.ncon, fpnlp.meta.nvar);
x2 = rand(fpnlp.meta.nvar);
@btime jac_coord!(fpnlp.nlp, x2, fpnlp.Ax.nzval)

nothing
#=
Iterative
[ Info:      3            Optml   4.0e-02   6.9e-09   2.7e-08   4.0e+00          Optimal   0.0e+00
 68.432747 seconds (128.60 M allocations: 6.658 GiB, 7.80% gc time, 41.55% compilation time)
  Counters:
             obj: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               grad: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 108               cons: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 114   
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 22    
           jprod: ███████████████████⋅ 333             jtprod: ████████████████████ 354               hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 44    
           hprod: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 50               jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  345.986 ns (3 allocations: 752 bytes)
  6.803 μs (42 allocations: 4.22 KiB)
  7.026 μs (45 allocations: 4.97 KiB)
  14.578 μs (79 allocations: 7.30 KiB)
  20.583 μs (135 allocations: 11.08 KiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 233448
           jprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 557155          jtprod: █████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 557155            hess: ███⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 233448
           hprod: ████████████████████ 2301784           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  11.581 μs (91 allocations: 6.14 KiB)
  9.142 μs (76 allocations: 4.86 KiB)
"Compare jacobian formation" = "Compare jacobian formation"
  907.122 ns (5 allocations: 592 bytes)
  2.490 μs (24 allocations: 2.75 KiB)

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

#=
ControlInvestment n = 1000
Iterative:
=#
#=
10.253 μs (6 allocations: 800 bytes)
  24.298 ms (2641 allocations: 84.01 MiB)
  25.348 ms (2645 allocations: 84.01 MiB)
  15.623 s (3354500 allocations: 22.36 GiB)
  3.282 s (515864 allocations: 12.28 GiB)
  Counters:
             obj: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                 cons: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 4     
           jprod: ████████████████████ 7894            jtprod: ████████████████████ 7894              hess: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 4     
           hprod: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1378             jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
  3.450 s (521033 allocations: 12.38 GiB)
  3.140 s (520511 allocations: 12.37 GiB)
"Compare jacobian formation" = "Compare jacobian formation"
  3.624 ms (511 allocations: 16.42 MiB)
  23.906 ms (535 allocations: 35.48 MiB)
=#