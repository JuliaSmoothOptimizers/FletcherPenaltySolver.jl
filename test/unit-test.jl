@testset "Unit test: FletcherPenaltyNLP" begin
    n = 10
    nlp = ADNLPModel(x->dot(x, x), zeros(n),
                     x->[sum(x) - 1.], zeros(1), zeros(1)) #ne second derivatives of the constraints
    demo_func = FletcherPenaltyNLPSolver._solve_system_dense
    fpnlp = FletcherPenaltyNLP(nlp)
    fpnlp = FletcherPenaltyNLP(nlp, sigma_0 = 0.5)
    fpnlp = FletcherPenaltyNLP(nlp, linear_system_solver = demo_func)
    fpnlp = FletcherPenaltyNLP(nlp, 0.5, demo_func)
    
    sigma = 0.5
    ys(x) = [(2-sigma)/n*sum(x) + sigma/n]
    Ys(x) = (2-sigma)/n * ones(n)
    
    @test fpnlp.meta.ncon == 0
    @test fpnlp.meta.nvar == 10
    
    @test isnan(fpnlp.fx)
    @test fpnlp.gx == Float64[]
    @test fpnlp.ys == Float64[]
    @test fpnlp.cx == Float64[]
    
    xfeas = ones(n)./n
    @test obj(fpnlp, xfeas) ≈ 0.1 atol = 1e-14

    @test fpnlp.fx  ≈ 0.1 atol = 1e-14
    @test fpnlp.gx ≈ 0.2*ones(n) atol = 1e-14
    @test fpnlp.ys ≈ ys(xfeas) atol = 1e-14
    @test fpnlp.cx ≈ [0.] atol = 1e-14
    
    @test grad(fpnlp, xfeas) ≈ zeros(n) atol = 1e-14
    
    @test Symmetric(hess(fpnlp, xfeas),:L) ≈ diagm(0 => 2. * ones(n)) - 2*ones(n)*Ys(xfeas)' atol = 1e-3
    vrand = rand(fpnlp.meta.nvar)
    @test hprod(fpnlp, xfeas, vrand ) ≈ 2*vrand - 2*ones(n)*Ys(xfeas)'*vrand atol = 1e-14
    
    xr = vcat(0.0, ones(9))
    cx = cons(nlp, xr)
    @test cx == [8.]
    
    @test obj(fpnlp, xr) ≈ (9. - dot(cx, ys(xr)))  atol = 1e-14
    @test grad(fpnlp, xr) ≈ 2*xr - Ys(xr).*cx - ones(n).*ys(xr) atol = 1e-14
    @test Symmetric(hess(fpnlp, xfeas),:L) ≈ diagm(0 => 2. * ones(n)) - 2*ones(n)*Ys(xfeas)' atol = 1e-3
    vrand = rand(fpnlp.meta.nvar)
    @test hprod(fpnlp, xr, vrand ) ≈ 2*vrand - 2*ones(n)*Ys(xfeas)'*vrand atol = 1e-14
    
    Is,Js = hess_structure(fpnlp)
    nnz = Int(fpnlp.meta.nvar * (fpnlp.meta.nvar + 1) / 2)
    @test fpnlp.meta.nnzh == nnz
    #Right now there are no sparse tries
    @test length(Is) == nnz
    @test length(Js) == nnz
    Vs = hess_coord(fpnlp, xfeas)
    @test length(Vs) == nnz
    _H = sparse(Is, Js, Vs)
    @test _H == hess(fpnlp, xfeas)
end
