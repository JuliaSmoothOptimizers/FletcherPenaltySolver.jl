#=
function multiple_precision_nlp(
  nlp :: AbstractNLPModel;
  precisions :: Array = [Float16, Float32, Float64, BigFloat],
  exclude = [ghjvprod])
  for T in precisions
    x = ones(T, nlp.meta.nvar)
    @test obj ∈ exclude || typeof(obj(nlp, x)) == T
    @test grad ∈ exclude || eltype(grad(nlp, x)) == T
    @test hess ∈ exclude || eltype(hess(nlp, x)) == T
    @test hess_op ∈ exclude || eltype(hess_op(nlp, x)) == T
    if hess_coord ∉ exclude && hess_op ∉ exclude
      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x)
      @test eltype(vals) == T
      Hv = zeros(T, nlp.meta.nvar)
      @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    end
    if nlp.meta.ncon > 0
      y = ones(T, nlp.meta.ncon)
      @test cons ∈ exclude || eltype(cons(nlp, x)) == T
      @test jac ∈ exclude || eltype(jac(nlp, x)) == T
      @test jac_op ∈ exclude || eltype(jac_op(nlp, x)) == T
      if jac_coord ∉ exclude && jac_op ∉ exclude
        rows, cols = jac_structure(nlp)
        vals = jac_coord(nlp, x)
        @test eltype(vals) == T
        Av = zeros(T, nlp.meta.ncon)
        Atv = zeros(T, nlp.meta.nvar)
        @test eltype(jac_op!(nlp, rows, cols, vals, Av, Atv)) == T
      end
      @test hess ∈ exclude || eltype(hess(nlp, x, y)) == T
      @test hess ∈ exclude || eltype(hess(nlp, x, y, obj_weight=one(T))) == T
      @test hess_op ∈ exclude || eltype(hess_op(nlp, x, y)) == T
      if hess_coord ∉ exclude && hess_op ∉ exclude
        rows, cols = hess_structure(nlp)
        vals = hess_coord(nlp, x, y)
        @test eltype(vals) == T
        Hv = zeros(T, nlp.meta.nvar)
        @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
      end
      @test ghjvprod ∈ exclude || eltype(ghjvprod(nlp, x, x, x)) == T
    end
  end
end
=#
@testset "NLP tests" begin
    nlp1 = ADNLPModel(x->dot(x, x), zeros(10), x->[sum(x) - 1.], zeros(1), zeros(1))
    demo_func = FletcherPenaltyNLPSolver._solve_system_dense
    problemset = [
        FletcherPenaltyNLP(nlp1, 0.5, demo_func, Val(1)),
        FletcherPenaltyNLP(nlp1, 0.5, demo_func, Val(2)),
    ]
    for nlp in problemset
      @testset "Problem $(nlp.meta.name)" begin
        @testset "Consistency" begin
          consistent_nlps([nlp, nlp])
        end
        @testset "Check dimensions" begin
          check_nlp_dimensions(nlp)
        end
        @testset "Multiple precision support" begin
          #multiple_precision_nlp(nlp) # not exactly working
        end
        @testset "View subarray" begin
          view_subarray_nlp(nlp) #https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/290 
        end
        @testset "Test coord memory" begin
          coord_memory_nlp(nlp)
        end
      end
    end
end
