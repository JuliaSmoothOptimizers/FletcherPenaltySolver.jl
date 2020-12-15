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
################################################################################
#using CUTEst, NLPModels, NLPModelsKnitro
#This package
#using FletcherPenaltyNLPSolver

@testset "Rank-deficient HS61" begin
   p     = "HS61"
   nlp   = CUTEstModel(p)
   lss   = Main.FletcherPenaltyNLPSolver._solve_with_linear_operator
   stats = Fletcher_penalty_solver(nlp, rtol = 1e-3,
                                   σ_0 = 1e3, ρ_0 = 1e3, δ_0 = 1e-2, 
                                   linear_system_solver = lss)
   @test stats.status == :first_order
   finalize(nlp)
end

@testset "Rank-deficient MSS1" begin
   p      = "MSS1"
   lss    = Main.FletcherPenaltyNLPSolver._solve_with_linear_operator
   nlp    = CUTEstModel(p)
   fpnlp  = FletcherPenaltyNLP(nlp, 1e3, 1e3, 1e-2, lss)
   stats1 = knitro(fpnlp)
   stats  = Fletcher_penalty_solver(nlp, rtol = 1e-3,
                                   σ_0 = 1e3, ρ_0 = 1e3, δ_0 = 1e-2, 
                                   linear_system_solver = lss)
   @test stats.status == :first_order
   finalize(nlp)
end
