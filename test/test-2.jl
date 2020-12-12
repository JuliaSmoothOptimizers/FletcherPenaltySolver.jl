
@testset "Rosenbrock with ∑x = 1" begin
    nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0],
                     x->[sum(x)-1], [0.0], [0.0])
    sol = [-1.612771347383541; 2.612771347383541]
    
    stats = with_logger(NullLogger()) do
      Fletcher_penalty_solver(nlp)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
    @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
    @test status == :first_order
  end

    @testset "Simple problem" begin
      n = 10
      nlp = ADNLPModel(x->dot(x, x), zeros(n),
                       x->[sum(x) - 1], zeros(1), zeros(1))
      #nlp.meta.x0 is an infeasible stationary point?

      stats = with_logger(NullLogger()) do
         Fletcher_penalty_solver(nlp)
      end
      x, dual, primal, status = stats.solution, stats.dual_feas, stats.primal_feas, stats.status
      @test norm(n * x - ones(n)) < 1e-6
      @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
      @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
      @test status == :first_order
    end

    @testset "HS6" begin
        nlp = ADNLPModel(x->(1 - x[1])^2, [-1.2; 1.0],
                         x->[10 * (x[2] - x[1]^2)], [0.0], [0.0])

        stats = with_logger(NullLogger()) do
          Fletcher_penalty_solver(nlp)
        end
        dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
        @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
        @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
        @test status == :first_order
      end

      @testset "HS7" begin
          nlp = ADNLPModel(x->log(1 + x[1]^2) - x[2], [2.0; 2.0],
                           x->[(1 + x[1]^2)^2 + x[2]^2 - 4], [0.0], [0.0])

          stats = with_logger(NullLogger()) do
            Fletcher_penalty_solver(nlp)
          end
          dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
          @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test status == :first_order
        end

        @testset "HS8" begin
          nlp = ADNLPModel(x->-1.0, [2.0; 1.0],
                           x->[x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 9], zeros(2), zeros(2))

          stats = with_logger(NullLogger()) do
            Fletcher_penalty_solver(nlp)
          end
          dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
          @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test status == :first_order
        end

        @testset "HS9" begin
          nlp = ADNLPModel(x->sin(π * x[1] / 12) * cos(π * x[2] / 16), zeros(2),
                           x->[4 * x[1] - 3 * x[2]], [0.0], [0.0])

          stats = with_logger(NullLogger()) do
            Fletcher_penalty_solver(nlp)
          end
          dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
          @test dual < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test status == :first_order
        end

        @testset "HS26" begin
          nlp = ADNLPModel(x->(x[1] - x[2])^2 + (x[2] - x[3])^4, [-2.6; 2.0; 2.0],
                           x->[(1 + x[2]^2) * x[1] + x[3]^4 - 3], [0.0], [0.0])
          stats = with_logger(NullLogger()) do
            Fletcher_penalty_solver(nlp)
          end
          dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
          @test dual < 1e-6  * max(norm(nlp.meta.x0), 1.)
          @test primal < 1e-6 * max(norm(nlp.meta.x0), 1.)
          @test status == :first_order
        end

        @testset "HS27" begin
          nlp = ADNLPModel(x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2, [2.0; 2.0; 2.0],
                           x->[x[1] + x[3]^2 + 1.0], [0.0], [0.0])
          stats = with_logger(NullLogger()) do
            Fletcher_penalty_solver(nlp)
          end
          dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
          @test dual < 1e-6  * max(norm(nlp.meta.x0), 1.)
          @test primal < 1e-6  * max(norm(nlp.meta.x0), 1.)
          @test status == :first_order
        end

        #
        # Huyer, W., & Neumaier, A. (2003). A new exact penalty function. SIAM Journal on Optimization, 13(4), 1141-1158.
        # Start from an infeasible stationary point.
        #
        @testset "Unbounded quadratic penalty" begin
            nlp = ADNLPModel(x->x[1]^3*x[2]^3, [0.0; 0.0],
                             x->[x[1]^2+x[2]^2-1], [0.0], [0.0])

            sol1 = [sqrt(2)/2; -sqrt(2)/2]
            sol2 = -sol1
            stats = with_logger(NullLogger()) do
              Fletcher_penalty_solver(nlp)
            end
            dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
            @test dual < 1e-6  * max(norm(nlp.meta.x0), 1.)
            @test primal < 1e-6  * max(norm(nlp.meta.x0), 1.)
            @test status == :first_order
          end
