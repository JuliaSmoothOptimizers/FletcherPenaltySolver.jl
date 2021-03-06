using LinearAlgebra, LinearOperators, SparseArrays

#JSO packages
using CUTEst, JSOSolvers, Logging, NLPModels, NLPModelsKnitro, NLPModelsIpopt, SolverTools

using Stopping

#For Test only
using Test

#This package
using FletcherPenaltyNLPSolver

# DIXCHLNG:  Knitro encounters an exception in evaluation callback: SingularException(3)
#            inv(A * A') was SingularException --- then one of the LDLFactorizations miss the regularization parameters
names = ["DIXCHLNG", "BT7", "FLT", "HS52", "HS47", "SPIN2OP"] #small problems first

nlp = CUTEstModel(names[1])
stats = fps_solve(nlp, max_iter = 100, hessian_approx = 2, 
                                     #linear_system_solve = FletcherPenaltyNLPSolver._solve_with_linear_operator,
                                     unconstrained_solver = knitro)
print(stats)
#=
finalize(nlp)

nlp = CUTEstModel(names[1])
meta = AlgoData(Float64)
x0, σ, ρ, δ = nlp.meta.x0, meta.σ_0, meta.ρ_0 , meta.δ_0
fp = FletcherPenaltyNLP(nlp, σ, ρ, δ, meta.linear_system_solver, meta.hessian_approx)
stats = knitro(fp)
=#
