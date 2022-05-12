@testset "Testing FPS-solver" begin
  @testset "$foo with 1st approximation" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
  ]
    foo(nlp -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(1)))
  end

  @testset "$foo with 2nd approximation" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
  ]
    foo(nlp -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(2)))
  end

  @testset "Multiprecision tests with 1st approximation" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(
        (nlp; kwargs...) -> fps_solve(
          nlp,
          nlp.meta.x0,
          hessian_approx = Val(1),
          unconstrained_solver = StoppingInterface.tron;
          kwargs...,
        ),
        ptype,
        precisions = (Float32, Float64, BigFloat),
      ) # precisions = (Float16, Float32, Float64, BigFloat)
    end
  end
  @testset "Multiprecision tests with 2nd approximation" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(
        (nlp; kwargs...) -> fps_solve(
          nlp,
          nlp.meta.x0,
          hessian_approx = Val(2),
          unconstrained_solver = StoppingInterface.tron;
          kwargs...,
        ),
        ptype,
        precisions = (Float32, Float64, BigFloat),
      ) # precisions = (Float16, Float32, Float64, BigFloat)
    end
  end
end
