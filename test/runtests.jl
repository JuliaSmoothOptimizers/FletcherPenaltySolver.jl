using LinearAlgebra, LinearOperators, Random, SparseArrays

#JSO packages
using ADNLPModels, Logging, NLPModels, NLPModelsTest, SolverCore, SolverTest
using JSOSolvers, NLPModelsIpopt, NLPModelsKnitro

Random.seed!(1234)

using Stopping, StoppingInterface

#For Test only
using Test

#This package
using FletcherPenaltySolver

@testset "Test callback" begin
  nlp = ADNLPModel(
    x -> (x[1] - x[2])^2 + (x[2] - x[3])^4,
    [-2.6; 2.0; 2.0],
    x -> [(1 + x[2]^2) * x[1] + x[3]^4 - 3],
    [0.0],
    [0.0],
  )
  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]
  function cb(nlp, solver, stats)
    x = stats.solution
    push!(X, x[1])
    push!(Y, x[2])
    if stats.iter == 4
      stats.status = :user
    end
  end
  stats = with_logger(NullLogger()) do
    fps_solve(nlp, σ_0 = 1.0, ρ_0 = 0.0, callback = cb)
  end
  @test stats.iter == 4
end

@testset "Test restart" begin
  include("restart.jl")
end

include("nlpmodelstest.jl")

@testset "Unit tests" begin
  include("unit-test.jl")
end

#Test the solvers:
mutable struct DummyModel{S, T} <: AbstractNLPModel{S, T}
  meta::NLPModelMeta{S, T}
  counters::Counters
end
@testset "Problem type error" begin
  nlp = DummyModel(NLPModelMeta{Float64, Vector{Float64}}(1, minimize = false), Counters())
  @test_throws ErrorException("fps_solve only works for minimization problem") fps_solve(
    nlp,
    zeros(1),
  )
  stp = NLPStopping(nlp)
  meta = FPSSSolver(stp, qds_solver = :iterative)
  stats = GenericExecutionStats(nlp)
  @test_throws ErrorException("fps_solve only works for minimization problem") SolverCore.solve!(
    meta,
    stp,
    stats,
  )
end

@testset "Test usual problems and difficult cases" begin
  @testset "Problem using KKT optimality" begin
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    c(x) = [x[1]^2 + x[2]^2 - 2]
    T = Float64
    x0 = T[-1.2; 1.0]
    ℓ, u = zeros(T, 2), 2 * ones(T, 2)
    nlp = ADNLPModel(f, x0, ℓ, u, c, zeros(1), zeros(1))
  
    ϵ = eps(T)^T(1 / 4)
  
    ng0 = norm(grad(nlp, nlp.meta.x0))
  
    stp = NLPStopping(nlp)
    stats = fps_solve(stp)
  
    @test eltype(stats.solution) == T
    @test stats.objective isa T
    @test stats.dual_feas isa T
    @test stats.primal_feas isa T
    @test stats.dual_feas < ϵ * ng0 + ϵ
    @test isapprox(stats.solution, ones(T, 2), atol = ϵ * ng0 * 10)
  end

  include("test-2.jl")
  include("rank-deficient.jl")
end

# Solver tests
include("solvertest.jl")
