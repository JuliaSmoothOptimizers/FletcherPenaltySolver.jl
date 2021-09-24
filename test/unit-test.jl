function test_memoization(fpnlp)
  #=
  xr = rand(fpnlp.meta.nvar)

  for f in [:obj, :grad, :objgrad]
    @test eval(f)(fpnlp, xr) != nothing
    tmp = (fpnlp.nlp.counters.neval_obj, fpnlp.nlp.counters.neval_grad, fpnlp.nlp.counters.neval_cons)
    @test eval(f)(fpnlp, xr) != nothing
    tmp2 = (fpnlp.nlp.counters.neval_obj, fpnlp.nlp.counters.neval_grad, fpnlp.nlp.counters.neval_cons)
    @test tmp == tmp2
  end
  =#

end

@testset "Unit test: FletcherPenaltyNLP with 1st hessian approximation" begin
  n = 10
  nlp = ADNLPModel(x -> dot(x, x), zeros(n), x -> [sum(x) - 1.0], zeros(1), zeros(1)) #ne second derivatives of the constraints
  fpnlp = FletcherPenaltyNLP(nlp)
  fpnlp = FletcherPenaltyNLP(nlp, σ_0 = 0.5)
  fpnlp = FletcherPenaltyNLP(nlp)
  fpnlp = FletcherPenaltyNLP(nlp, 0.5, Val(1))

  @test fpnlp.hessian_approx == Val(1)

  σ = 0.5
  ys(x) = [(2 - σ) / n * sum(x) + σ / n]
  Ys(x) = (2 - σ) / n * ones(n)

  @test fpnlp.meta.ncon == 0
  @test fpnlp.meta.nvar == 10

  @test isnan(fpnlp.fx)
  @test length(fpnlp.gx) == nlp.meta.nvar
  @test length(fpnlp.ys) == nlp.meta.ncon
  @test length(fpnlp.cx) == nlp.meta.ncon

  test_memoization(fpnlp)

  xfeas = ones(n) ./ n
  @test obj(fpnlp, xfeas) ≈ 0.1 atol = 1e-14

  @test fpnlp.fx ≈ 0.1 atol = 1e-14
  @test fpnlp.gx ≈ 0.2 * ones(n) atol = 1e-14
  @test fpnlp.ys ≈ ys(xfeas) atol = 1e-14
  @test fpnlp.cx ≈ [0.0] atol = 1e-14

  @test grad(fpnlp, xfeas) ≈ zeros(n) atol = 1e-14

  #@test hess(fpnlp, xfeas) ≈ diagm(0 => 2. * ones(n)) - 2*ones(n)*Ys(xfeas)' atol = 1e-3
  vrand = rand(fpnlp.meta.nvar)
  @test hprod(fpnlp, xfeas, vrand) ≈ hess(fpnlp, xfeas) * vrand atol = 1e-12 #2*vrand - 2*ones(n)*Ys(xfeas)'*vrand atol = 1e-14

  xr = vcat(0.0, ones(9))
  cx = cons(nlp, xr)
  @test cx == [8.0]

  @test obj(fpnlp, xr) ≈ (9.0 - dot(cx, ys(xr))) atol = 1e-13
  @test grad(fpnlp, xr) ≈ 2 * xr - Ys(xr) .* cx - ones(n) .* ys(xr) atol = 1e-13

  #@test hess(fpnlp, xfeas) ≈ diagm(0 => 2. * ones(n)) - 2*ones(n)*Ys(xfeas)' atol = 1e-3
  vrand = rand(fpnlp.meta.nvar)
  #@test hprod(fpnlp, xr, vrand ) ≈ 2*vrand - 2*ones(n)*Ys(xfeas)'*vrand atol = 1e-14
  @test hprod(fpnlp, xr, vrand) ≈ hess(fpnlp, xr) * vrand atol = 1e-12

  Is, Js = hess_structure(fpnlp)
  nnz = Int(fpnlp.meta.nvar * (fpnlp.meta.nvar + 1) / 2)
  @test fpnlp.meta.nnzh == nnz
  #Right now there are no sparse tries
  @test length(Is) == nnz
  @test length(Js) == nnz
  Vs = hess_coord(fpnlp, xfeas)
  @test length(Vs) == nnz
  _H = sparse(Is, Js, Vs)
  @test _H == hess(fpnlp, xfeas).data
end

@testset "Unit test: FletcherPenaltyNLP with 1st hessian approximation and regularization" begin
  nlp = ADNLPModel(
    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
    zeros(2),
    x -> [x[1]^2 + x[2]^2 - 1],
    zeros(1),
    zeros(1),
  )
  @test equality_constrained(nlp)

  fpnlp = FletcherPenaltyNLP(nlp, 0.5, 0.1, 0.25, Val(1))

  @test fpnlp.meta.ncon == 0
  @test fpnlp.meta.nvar == 2

  @test fpnlp.σ == 0.5
  @test fpnlp.ρ == 0.1
  @test fpnlp.δ == 0.25
  σ, ρ, δ = fpnlp.σ, fpnlp.ρ, fpnlp.δ

  @test fpnlp.hessian_approx == Val(1)

  xr = [sqrt(6) / 3; sqrt(3) / 3]

  D(x) = -(4 * x[1]^2 + 4 * x[2]^2 + δ)
  ys(x) =
    (
      2 * x[1] * (-2 * (x[1] - 1) + 400 * x[1] * (x[2] - x[1]^2)) - 400 * x[2] * (x[2] - x[1]^2) +
      σ * (x[1]^2 + x[2]^2 - 1)
    ) / D(x)
  Ys(x) = ForwardDiff.gradient(ys, x)
  DYs(x) = ForwardDiff.hessian(ys, x)

  @test isnan(fpnlp.fx)
  @test length(fpnlp.gx) == nlp.meta.nvar
  @test length(fpnlp.ys) == nlp.meta.ncon
  @test length(fpnlp.cx) == nlp.meta.ncon

  test_memoization(fpnlp)

  @test obj(fpnlp, xr) ≈ (sqrt(6) - 3)^2 / 9 + 100 * (sqrt(3) - 2)^2 / 9 atol = 1e-14

  @test fpnlp.fx ≈ (sqrt(6) - 3)^2 / 9 + 100 * (sqrt(3) - 2)^2 / 9 atol = 1e-14
  @test fpnlp.gx ≈ [
    2 * (sqrt(6) / 3 - 1) - 400 * sqrt(6) / 3 * (sqrt(3) / 3 - 6 / 9)
    200 * (sqrt(3) / 3 - 6 / 9)
  ] atol = 1e-13
  @test fpnlp.ys ≈ [ys(xr)] atol = 1e-14
  @test fpnlp.cx ≈ [0.0] atol = 1e-14

  @test objgrad(fpnlp, xr)[1] ≈ obj(fpnlp, xr)
  @test objgrad(fpnlp, xr)[2] ≈ grad(fpnlp, xr)

  vrand = rand(fpnlp.meta.nvar)
  @test hprod(fpnlp, xr, vrand) ≈ hess(fpnlp, xr) * vrand atol = 1e-12

  #=
  #Tanj: note that we use here an approximation of the hessian matrix
  H(x) = [2-100*4*x[2]+100*12*x[1]^2 -400*x[1]; -400*x[1] 200]
  C(x) = [2*x[1];2*x[2]]
  DC(x) = [2 0; 0 2]
  G(x) = (grad(nlp, x) - C(x)*ys(x)-(x[1]^2 + x[2]^2 - 1)*Ys(x)+ρ*(x[1]^2 + x[2]^2 - 1)*C(x))
  @test grad(fpnlp, xr) ≈  G(xr) atol = 1e-13
  function Hess(x)
      Lag = H(x) - (ys(x)-ρ*(x[1]^2 + x[2]^2 - 1)) * DC(x)
      reg = ρ*C(x)*C(x)'
     return Lag + reg- Ys(x)*C(x)' # - (x[1]^2 + x[2]^2 - 1)*DYs(x)  - C(x)*Ys(x)'
  end
  #Hess(xr) - ForwardDiff.jacobian(G, xr)
  vr   = rand(nlp.meta.nvar)
  Hxv  = Hess(xr) * vr
  _Hxv = hprod(fpnlp, xr, vr)
  @show Hxv - _Hxv
  =#
end

@testset "Unit test: FletcherPenaltyNLP with 2nd hessian approximation" begin
  n = 10
  nlp = ADNLPModel(x -> dot(x, x), zeros(n), x -> [sum(x) - 1.0], zeros(1), zeros(1)) #ne second derivatives of the constraints
  fpnlp = FletcherPenaltyNLP(nlp)
  fpnlp = FletcherPenaltyNLP(nlp, σ_0 = 0.5)
  fpnlp = FletcherPenaltyNLP(nlp)
  fpnlp = FletcherPenaltyNLP(nlp, 0.5, Val(2))

  @test fpnlp.hessian_approx == Val(2)

  σ = 0.5
  ys(x) = [(2 - σ) / n * sum(x) + σ / n]
  Ys(x) = (2 - σ) / n * ones(n)

  @test fpnlp.meta.ncon == 0
  @test fpnlp.meta.nvar == 10

  @test isnan(fpnlp.fx)
  @test length(fpnlp.gx) == nlp.meta.nvar
  @test length(fpnlp.ys) == nlp.meta.ncon
  @test length(fpnlp.cx) == nlp.meta.ncon

  test_memoization(fpnlp)

  xfeas = ones(n) ./ n
  @test obj(fpnlp, xfeas) ≈ 0.1 atol = 1e-14

  @test fpnlp.fx ≈ 0.1 atol = 1e-14
  @test fpnlp.gx ≈ 0.2 * ones(n) atol = 1e-14
  @test fpnlp.ys ≈ ys(xfeas) atol = 1e-14
  @test fpnlp.cx ≈ [0.0] atol = 1e-14

  @test grad(fpnlp, xfeas) ≈ zeros(n) atol = 1e-14

  @test hess(fpnlp, xfeas) ≈ diagm(0 => 2.0 * ones(n)) - 2 * ones(n) * Ys(xfeas)' atol = 1e-3
  vrand = rand(fpnlp.meta.nvar)
  @test hprod(fpnlp, xfeas, vrand) ≈ hess(fpnlp, xfeas) * vrand atol = 1e-13
  @test hprod(fpnlp, xfeas, vrand) ≈ 2 * vrand - 2 * ones(n) * Ys(xfeas)' * vrand atol = 1e-13

  xr = vcat(0.0, ones(9))
  cx = cons(nlp, xr)
  @test cx == [8.0]

  @test obj(fpnlp, xr) ≈ (9.0 - dot(cx, ys(xr))) atol = 1e-14
  @test grad(fpnlp, xr) ≈ 2 * xr - Ys(xr) .* cx - ones(n) .* ys(xr) atol = 1e-14
  @test hess(fpnlp, xfeas) ≈ diagm(0 => 2.0 * ones(n)) - 2 * ones(n) * Ys(xfeas)' atol = 1e-3
  vrand = rand(fpnlp.meta.nvar)
  @test hprod(fpnlp, xr, vrand) ≈ hess(fpnlp, xr) * vrand atol = 1e-12
  @test hprod(fpnlp, xr, vrand) ≈ 2 * vrand - 2 * ones(n) * Ys(xfeas)' * vrand atol = 1e-12

  Is, Js = hess_structure(fpnlp)
  nnz = Int(fpnlp.meta.nvar * (fpnlp.meta.nvar + 1) / 2)
  @test fpnlp.meta.nnzh == nnz
  #Right now there are no sparse tries
  @test length(Is) == nnz
  @test length(Js) == nnz
  Vs = hess_coord(fpnlp, xfeas)
  @test length(Vs) == nnz
  _H = sparse(Is, Js, Vs)
  @test _H == hess(fpnlp, xfeas).data
end

@testset "Unit test: FletcherPenaltyNLP with 2nd hessian approximation and regularization" begin
  nlp = ADNLPModel(
    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
    zeros(2),
    x -> [x[1]^2 + x[2]^2 - 1],
    zeros(1),
    zeros(1),
  )
  @test equality_constrained(nlp)

  fpnlp = FletcherPenaltyNLP(nlp, 0.5, 0.1, 0.25, Val(2))

  @test fpnlp.meta.ncon == 0
  @test fpnlp.meta.nvar == 2

  @test fpnlp.σ == 0.5
  @test fpnlp.ρ == 0.1
  @test fpnlp.δ == 0.25
  σ, ρ, δ = fpnlp.σ, fpnlp.ρ, fpnlp.δ

  @test fpnlp.hessian_approx == Val(2)

  xr = [sqrt(6) / 3; sqrt(3) / 3]

  D(x) = -(4 * x[1]^2 + 4 * x[2]^2 + δ)
  ys(x) =
    (
      2 * x[1] * (-2 * (x[1] - 1) + 400 * x[1] * (x[2] - x[1]^2)) - 400 * x[2] * (x[2] - x[1]^2) +
      σ * (x[1]^2 + x[2]^2 - 1)
    ) / D(x)
  Ys(x) = ForwardDiff.gradient(ys, x)
  DYs(x) = ForwardDiff.hessian(ys, x)

  @test isnan(fpnlp.fx)
  @test length(fpnlp.gx) == nlp.meta.nvar
  @test length(fpnlp.ys) == nlp.meta.ncon
  @test length(fpnlp.cx) == nlp.meta.ncon

  test_memoization(fpnlp)

  @test obj(fpnlp, xr) ≈ (sqrt(6) - 3)^2 / 9 + 100 * (sqrt(3) - 2)^2 / 9 atol = 1e-14

  @test fpnlp.fx ≈ (sqrt(6) - 3)^2 / 9 + 100 * (sqrt(3) - 2)^2 / 9 atol = 1e-14
  @test fpnlp.gx ≈ [
    2 * (sqrt(6) / 3 - 1) - 400 * sqrt(6) / 3 * (sqrt(3) / 3 - 6 / 9)
    200 * (sqrt(3) / 3 - 6 / 9)
  ] atol = 1e-13
  @test fpnlp.ys ≈ [ys(xr)] atol = 1e-14
  @test fpnlp.cx ≈ [0.0] atol = 1e-14

  @test objgrad(fpnlp, xr)[1] ≈ obj(fpnlp, xr)
  @test objgrad(fpnlp, xr)[2] ≈ grad(fpnlp, xr)

  #=
  #Tanj: note that we use here an approximation of the hessian matrix
  H(x) = [2-100*4*x[2]+100*12*x[1]^2 -400*x[1]; -400*x[1] 200]
  C(x) = [2*x[1];2*x[2]]
  DC(x) = [2 0; 0 2]
  G(x) = (grad(nlp, x) - C(x)*ys(x)-(x[1]^2 + x[2]^2 - 1)*Ys(x)+ρ*(x[1]^2 + x[2]^2 - 1)*C(x))
  @test grad(fpnlp, xr) ≈  G(xr) atol = 1e-13
  function Hess(x)
      Lag = H(x) - (ys(x)-ρ*(x[1]^2 + x[2]^2 - 1)) * DC(x)
      reg = ρ*C(x)*C(x)'
     return Lag + reg- Ys(x)*C(x)' # - (x[1]^2 + x[2]^2 - 1)*DYs(x)  - C(x)*Ys(x)'
  end
  #Hess(xr) - ForwardDiff.jacobian(G, xr)
  vr   = rand(nlp.meta.nvar)
  Hxv  = Hess(xr) * vr
  _Hxv = hprod(fpnlp, xr, vr)
  @show Hxv - _Hxv
  =#
end
