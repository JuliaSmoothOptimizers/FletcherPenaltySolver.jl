using LinearAlgebra, LinearOperators, Random, SparseArrays

#JSO packages
using CUTEst, JSOSolvers, Logging, NLPModels, NLPModelsIpopt, SolverTools
using ADNLPModels, NLPModelsTest, SolverTest

Random.seed!(1234)

using Stopping

#For Test only
using Test

#This package
using FletcherPenaltyNLPSolver

include("nlpmodelstest.jl")

include("unit-test.jl")
@warn "test_double_linear_algebra.jl: To be replaced"
#include("test_double_linear_algebra.jl")
#Test the solvers:
mutable struct DummyModel <: AbstractNLPModel
  meta::NLPModelMeta
  counters::Counters
end
@testset "Problem type error" begin
  nlp = DummyModel(NLPModelMeta(1, minimize = false), Counters())
  @test_throws ErrorException("fps_solve only works for minimization problem") fps_solve(
    nlp,
    zeros(1),
  )
  stp = NLPStopping(nlp)
  meta = FPSSSolver(nlp, 0.0)
  @test_throws ErrorException("fps_solve only works for minimization problem") fps_solve(stp, meta)
end
#On a toy rosenbrock variation.
include("test-0.jl")
#On a problem from the package OptimizationProblems
include("test-1.jl")
#On usual test problems
include("test-2.jl")

#Rank-deficient problems
include("rank-deficient.jl")

# Solver tests
# include("solvertest.jl")
