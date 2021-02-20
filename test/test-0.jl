@testset "simple problem" begin
    
    n = 2^4
    c(x) = [sum(x[findall( x->mod(x,2)==0, 1:n)]) ] #c(x) = [x[2] + x[4] + x[6]]
    lcon, ucon = zeros(1), zeros(1)
    sos = ADNLPModel(x->sum((x .- 1).^2), zeros(n), c, lcon, ucon)
    σ_0 = 0.1
    #fp_sos0 = FletcherPenaltyNLP(NLPModelMeta(n), Counters(), sos, σ_0, _solve_system_dense, 2)
    fp_sos  = FletcherPenaltyNLP(sos, σ_0, FletcherPenaltyNLPSolver._solve_with_linear_operator, 2)
    fp_sos2 = FletcherPenaltyNLP(sos, σ_0, FletcherPenaltyNLPSolver._solve_system_factorization_lu, 2)
    solx = zeros(n); solx[findall( x->mod(x,2)==1, 1:n)] = ones(Int(n/2))

    #In the paper, they use a Newton method.
    x0 = zeros(n)
    @time x1,f1,g1,H1 = lbfgs(fp_sos, x0, lsfunc = FletcherPenaltyNLPSolver.armijo_og)
    fp_sos.nlp.counters
    reset!(fp_sos.nlp)
    @time x2,f2,g2,H2 = lbfgs(fp_sos, x0, lsfunc = SolverTools.armijo_wolfe)
    fp_sos.nlp.counters

    @time x3,f3,g3,H3 = lbfgs(fp_sos2, x0, lsfunc = FletcherPenaltyNLPSolver.armijo_og)
    fp_sos2.nlp.counters
    reset!(fp_sos2.nlp)
    @time x4,f4,g4,H4 = lbfgs(fp_sos2, x0, lsfunc = SolverTools.armijo_wolfe)
    fp_sos2.nlp.counters

    @test norm(x1 - solx) <= sqrt(eps(Float64))
    @test norm(x2 - solx) <= sqrt(eps(Float64))
    @test norm(x3 - solx) <= sqrt(eps(Float64))
    @test norm(x4 - solx) <= sqrt(eps(Float64))

    #the 2nd try seems to be the fastest.
    fp_sos_stp = NLPStopping(fp_sos, optimality_check = unconstrained_check, atol = 1e-3)
    @time fp_sos_stp = lbfgs(fp_sos_stp, x0 = x0, lsfunc = SolverTools.armijo_wolfe)
    @test norm(fp_sos_stp.current_state.x - x2) <= sqrt(eps(Float64))

    #sos_stp = NLPStopping(sos, optimality_check = unconstrained_check)
    stats = Fletcher_penalty_solver(sos, sos.meta.x0)

    #Test de compilation:
    hprod(fp_sos, rand(fp_sos.meta.nvar),rand(fp_sos.meta.nvar))
    hess(fp_sos, rand(fp_sos.meta.nvar))
    @test hprod(fp_sos, rand(fp_sos.meta.nvar),zeros(fp_sos.meta.nvar)) == zeros(fp_sos.meta.nvar)

    #using NLPModelsIpopt
    #@warn "Bad idea, it is very slow" 
    #ipopt(fp_sos)

end
