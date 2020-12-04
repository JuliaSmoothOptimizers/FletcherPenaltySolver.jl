using LinearAlgebra, LinearOperators, SparseArrays

#JSO packages
using CUTEst, JSOSolvers, NLPModels, SolverTools

using Stopping

#For Test only
using Test

#This package
using FletcherPenaltyNLPSolver

include("unit-test.jl")
#Test the solvers:
#On a toy rosenbrock variation.
include("test-0.jl")
#On a problem from the package OptimizationProblems
include("test-1.jl")
#On a problem from the package CUTEst
#include("test-2.jl")
