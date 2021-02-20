using Test

#include("../src/FletcherPenaltyNLPSolver.jl")
#using Main.FletcherPenaltyNLPSolver

using OptimizationProblems, NLPModelsJuMP

@testset "sbrybnd" begin
    _model = sbrybnd()

    nlp = MathOptNLPModel(_model)
    n, x0 = nlp.meta.nvar, nlp.meta.x0

    σ_0 = 0.1

    fp_sos  = FletcherPenaltyNLP(nlp, σ_0, FletcherPenaltyNLPSolver._solve_with_linear_operator, 2)
    fp_sos2 = FletcherPenaltyNLP(nlp, σ_0, FletcherPenaltyNLPSolver._solve_system_factorization_lu, 2)

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

    @test obj(nlp, x1) == obj(nlp, x2)
    @test obj(nlp, x1) == obj(nlp, x3)
    @test obj(nlp, x1) == obj(nlp, x4)

    fp_sos_stp = NLPStopping(fp_sos, optimality_check = unconstrained_check, atol = 1e-3)
    @time fp_sos_stp = lbfgs(fp_sos_stp, x0 = x0, lsfunc = SolverTools.armijo_wolfe)
    @test norm(fp_sos_stp.current_state.x - x2) <= sqrt(eps(Float64))

    #sos_stp = NLPStopping(sos, optimality_check = unconstrained_check)
    stats = Fletcher_penalty_solver(nlp, nlp.meta.x0, max_iter = 10) #status(stp) = :ResourcesExhausted

end
