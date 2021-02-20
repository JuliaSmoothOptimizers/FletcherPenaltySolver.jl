using LinearAlgebra, LinearOperators, SparseArrays

#JSO packages
using CUTEst, JSOSolvers, Logging, NLPModels, NLPModelsKnitro, NLPModelsIpopt, SolverTools

using Stopping

#For Test only
using Test

#This package
using FletcherPenaltyNLPSolver

include("unit-test.jl")
include("test_double_linear_algebra.jl")
#Test the solvers:
#On a toy rosenbrock variation.
include("test-0.jl")
#On a problem from the package OptimizationProblems
include("test-1.jl")
#On problems using KNITRO
include("test-2.jl")

#Rank-deficient problems
#include("rank-deficient.jl")
