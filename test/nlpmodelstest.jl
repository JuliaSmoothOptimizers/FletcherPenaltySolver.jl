function test_fps_model(T, σ, vivi, qds_type)
  nlp1 = ADNLPModel(x -> dot(x, x), zeros(T, 10), x -> [sum(x) - 1.0], zeros(T, 1), zeros(T, 1))
  qds = FletcherPenaltyNLPSolver.eval(qds_type)(nlp1, T(0))
  return FletcherPenaltyNLP(nlp1, T(σ), vivi, qds = qds)
end

function test_fps_lin_model(T, σ, vivi, qds_type)
  nlp1 = ADNLPModel(x -> dot(x, x), zeros(T, 10), sparse(ones(T, 1, 10)), x -> [sum(x)], zeros(T, 2), zeros(T, 2))
  qds = FletcherPenaltyNLPSolver.eval(qds_type)(nlp1, T(0))
  return FletcherPenaltyNLP(nlp1, T(σ), vivi, qds = qds)
end

@testset "NLP tests" begin
  problemset = [
    (T = Float64) -> test_fps_model(T, 0.5, Val(1), :LDLtSolver),
    (T = Float64) -> test_fps_model(T, 0.5, Val(2), :LDLtSolver),
    (T = Float64) -> test_fps_model(T, 0.5, Val(1), :IterativeSolver),
    (T = Float64) -> test_fps_model(T, 0.5, Val(2), :IterativeSolver),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(1), :LDLtSolver),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(2), :LDLtSolver),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(1), :IterativeSolver),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(2), :IterativeSolver),
  ]
  for nlp_from_T in problemset
    nlp = nlp_from_T()
    @testset "Problem $(nlp.meta.name)" begin
      @testset "Consistency" begin
        consistent_nlps([nlp, nlp], linear_api = true)
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp, linear_api = true)
      end
      @testset "Multiple precision support" begin
        multiple_precision_nlp(nlp_from_T, linear_api = true)
      end
      @testset "View subarray" begin
        view_subarray_nlp(nlp)
      end
      @testset "Test coord memory" begin
        coord_memory_nlp(nlp, linear_api = true)
      end
    end
  end
end
