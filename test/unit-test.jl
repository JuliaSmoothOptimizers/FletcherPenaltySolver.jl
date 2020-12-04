@testset "Unit test: FletcherPenaltyNLP" begin
    n = 10
    nlp = ADNLPModel(x->dot(x, x), zeros(n),
                     x->[sum(x) - 1.], zeros(1), zeros(1))
    demo_func = FletcherPenaltyNLPSolver._solve_system_dense
    fpnlp = FletcherPenaltyNLP(nlp)
    fpnlp = FletcherPenaltyNLP(nlp, sigma_0 = 0.5)
    fpnlp = FletcherPenaltyNLP(nlp, linear_system_solver = demo_func)
    fpnlp = FletcherPenaltyNLP(nlp, 0.5, demo_func)
    
    @test fpnlp.meta.ncon == 0
    @test fpnlp.meta.nvar == 10
    
    xfeas = ones(n)./10.
    @test obj(fpnlp, xfeas) ≈ 0.1 atol = 1e-14
end
