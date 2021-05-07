#using LinearAlgebra, SparseArrays, NLPModels, Test
#This packages
#using FletcherPenaltyNLPSolver    

@testset "simple problem" begin
  function _test_double_system(r::T, nlp) where {T}
    sol1 = rand(T, nlp.meta.nvar)
    nlp_fps = FletcherPenaltyNLP(nlp)
    rhs1 = rand(T, nlp.meta.nvar + nlp.meta.ncon)
    rhs2 = rand(T, nlp.meta.nvar + nlp.meta.ncon)

    fcts = [
      :_solve_ldlt_factorization,
      :_solve_with_linear_operator,
      :_solve_system_dense,
      :_solve_system_factorization_lu,
    ]

    nlp_fps.δ = 1.0

    for func ∈ fcts
      s11, s12 = FletcherPenaltyNLPSolver.eval(func)(nlp_fps, sol1, rhs1, nothing)
      s21, s22 = FletcherPenaltyNLPSolver.eval(func)(nlp_fps, sol1, rhs1, rhs2)
      @test s11 == s21
      @test s21[1:(nlp.meta.nvar)] +
            jtprod(nlp, sol1, s21[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)]) ≈
            rhs1[1:(nlp.meta.nvar)]
      @test jprod(nlp, sol1, s21[1:(nlp.meta.nvar)]) -
            nlp_fps.δ * s21[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)] ≈
            rhs1[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)]
      @test isnothing(s12)
      @test s22[1:(nlp.meta.nvar)] +
            jtprod(nlp, sol1, s22[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)]) ≈
            rhs2[1:(nlp.meta.nvar)]
      @test jprod(nlp, sol1, s22[1:(nlp.meta.nvar)]) -
            nlp_fps.δ * s22[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)] ≈
            rhs2[(nlp.meta.nvar + 1):(nlp.meta.nvar + nlp.meta.ncon)]
      #test also the return type
    end
  end

  nlp = ADNLPModel(x -> 0.0, zeros(1), x -> [x[1]^3 + x[1] - 2.0], zeros(1), zeros(1))

  _test_double_system(1.0, nlp)

  nlp = ADNLPModel(
    x -> -1.0,
    [2.0; 1.0],
    x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 9],
    zeros(2),
    zeros(2),
  )

  _test_double_system(1.0, nlp)
end
