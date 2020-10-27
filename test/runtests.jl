using ForwardDiff, Gridap, JSOSolvers, LinearAlgebra, NLPModels, Test

include("../src/FletcherPenaltyNLPSolver.jl")

#Test the solvers:
#On a toy rosenbrock variation.
include("test-0.jl")
#On a problem from the package OptimizationProblems
include("test-1.jl")
#On a problem from the package CUTEst
#include("test-2.jl")
