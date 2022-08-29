function test_fps_model(T, σ, vivi, qds_type)
  nlp1 = ADNLPModel(x -> dot(x, x), zeros(T, 10), x -> [sum(x)], ones(T, 1), ones(T, 1))
  qds = FletcherPenaltySolver.eval(qds_type)(nlp1, T(0))
  return FletcherPenaltyNLP(nlp1, T(σ), vivi, qds = qds)
end

function test_fps_lin_model(T, σ, vivi, qds_type, use_linear = false)
  nvar = 10
  nlp1 =
    ADNLPModel(x -> dot(x, x), zeros(T, nvar), sparse(ones(T, 1, nvar)), ones(T, 1), ones(T, 1))
  qds =
    FletcherPenaltySolver.eval(qds_type)(nlp1, T(0), explicit_linear_constraints = use_linear)
  return FletcherPenaltyNLP(nlp1, T(σ), vivi, qds = qds, explicit_linear_constraints = use_linear)
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
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(1), :LDLtSolver, true),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(2), :LDLtSolver, true),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(1), :IterativeSolver, true),
    (T = Float64) -> test_fps_lin_model(T, 0.5, Val(2), :IterativeSolver, true),
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
        multiple_precision_nlp(
          nlp_from_T,
          linear_api = true,
          precisions = [Float16, Float32, Float64],
        )
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
