################################################################################
#
# December 15, 2020: T.M.
# This is a test on rank-deficient jacobian problems, following the discussion
# in Section 9.4 of Estrin et al. (2020).
#
# Works well :).
#
# Comments:
# Convergence of x is O(δ), so it should be small or converge to a small value.
# How much does it help the least-square?
#
# Warning: ghjvprod and jth_hess not implemented for CUTEstModel,
# so hessian_approx = 1 cannot be used.
#
################################################################################
#using CUTEst, NLPModels, NLPModelsKnitro
#This package
#using FletcherPenaltyNLPSolver

@testset "Rank-deficient HS61" begin
  p     = "HS61"
  nlp   = CUTEstModel(p)
  lss   = FletcherPenaltyNLPSolver._solve_with_linear_operator
#=
  stats = fps_solve(nlp, nlp.meta.x0, 
                                  σ_0 = 1e3, ρ_0 = 1e3, δ_0 = 1e-2, 
                                  linear_system_solver = lss,
                                  hessian_approx = 1, #error with hessian_approx = 1
                                  rtol = 1e-3)
  @test stats.status == :first_order
=#
  stats = fps_solve(nlp, nlp.meta.x0, 
                                  σ_0 = 1e3, ρ_0 = 1e3, δ_0 = 1e-2, 
                                  linear_system_solver = lss,
                                  hessian_approx = 2, #error with hessian_approx = 1
                                  rtol = 1e-3)
  @test stats.status == :first_order
  finalize(nlp)
end

@testset "Rank-deficient MSS1" begin
  p      = "MSS1"
  nlp    = CUTEstModel(p)
  lss    = FletcherPenaltyNLPSolver._solve_with_linear_operator
  fpnlp  = FletcherPenaltyNLP(nlp, 1e3, 1e3, 1e-2, lss, 2)
  stats1 = ipopt(fpnlp, print_level = 0) #knitro
  stats  = fps_solve(nlp, nlp.meta.x0, rtol = 1e-3,
                                   σ_0 = 1e3, ρ_0 = 1e3, δ_0 = 1e-2, 
                                   linear_system_solver = lss,
                                   hessian_approx = 2) #error with hessian_approx = 1
  @test stats.status == :first_order
  finalize(nlp)
end
