import NLPModels:
  increment!,
  obj,
  objgrad,
  objgrad!,
  grad!,
  grad,
  hess,
  hprod,
  hprod!,
  hess_coord,
  hess_coord!,
  hess_structure,
  hess_structure!
"""
We consider here the implementation of Fletcher's exact penalty method for
the minimization problem:

    minₓ f(x) s.t. c(x) = 0

using Fletcher penalty function:
    
    minₓ f(x) - dot(c(x),ys(x)) + ρ/2 dot(c(x),c(x))

where

    ys(x) := argmin\\_y 0.5 ||A(x)y - g(x)||²₂ + σ c(x)^T y + 0.5 δ ||²₂

and denote Ys the gradient of ys(x).

`FletcherPenaltyNLP(:: AbstractNLPModel, :: Number, :: Function)`
or
`FletcherPenaltyNLP(:: AbstractNLPModel; σ_0 :: Real = 1.0, linear_system_solver :: Function = _solve_with_linear_operator)`

Notes:
- Evaluation of the obj, grad, objgrad functions evaluate functions from the orginial nlp.
These values are stored in `fx`, `cx`, `gx`.
- The value of the penalty vector `ys` is also stored.
- `linear_system_solver(nlp, x, rhs1, Union{rhs2,nothing})` is a function that successively solve
the two linear systems and returns the two solutions.

TODO:
- sparse structure of the hessian?
- avoid reevaluation fx, cx, gs, ys?

Example:
fp_sos  = FletcherPenaltyNLP(nlp, 0.1, _solve_with_linear_operator)
"""
mutable struct FletcherPenaltyNLP{
  S <: AbstractFloat,
  T <: AbstractVector{S},
  A <: Union{Val{1}, Val{2}},
} <: AbstractNLPModel
  meta::AbstractNLPModelMeta
  counters::Counters
  nlp::AbstractNLPModel

  #Evaluation of the FletcherPenaltyNLP functions contains info on nlp:
  fx::Union{S}
  cx::Union{T}
  gx::Union{T}
  ys::Union{T}

  σ::Real
  ρ::Real
  δ::Real
  linear_system_solver::Function

  hessian_approx::A
end

function FletcherPenaltyNLP(nlp, σ, linear_system_solver, hessian_approx; x0 = nlp.meta.x0)
  S, T = eltype(x0), typeof(x0)

  nvar = nlp.meta.nvar

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    nnzh = nnzh,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
  )
  counters = Counters()
  return FletcherPenaltyNLP(
    meta,
    counters,
    nlp,
    S(NaN),
    S[],
    S[],
    S[],
    σ,
    0.0,
    0.0,
    linear_system_solver,
    hessian_approx,
  )
end

function FletcherPenaltyNLP(nlp, σ, ρ, δ, linear_system_solver, hessian_approx; x0 = nlp.meta.x0)
  S, T = eltype(x0), typeof(x0)

  nvar = nlp.meta.nvar

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    nnzh = nnzh,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
  )
  counters = Counters()
  return FletcherPenaltyNLP(
    meta,
    counters,
    nlp,
    S(NaN),
    S[],
    S[],
    S[],
    σ,
    ρ,
    δ,
    linear_system_solver,
    hessian_approx,
  )
end

#Set of functions solving two linear systems with different rhs.
# linear_system_solver(nlp, x, rhs1, rhs2; kwargs...)
# List of implemented methods:
# i)   _solve_system_dense
# ii)  _solve_with_linear_operator
# iii) _solve_system_factorization_eigenvalue
# iv)  _solve_system_factorization_lu
include("solve_two_systems.jl")

include("linesearch.jl")

function FletcherPenaltyNLP(
  nlp::AbstractNLPModel;
  σ_0::Real = one(eltype(nlp.meta.x0)),
  rho_0::Real = zero(eltype(nlp.meta.x0)),
  delta_0::Real = zero(eltype(nlp.meta.x0)),
  linear_system_solver::Function = _solve_with_linear_operator,
  hessian_approx = Val(2),
  x0 = nlp.meta.x0,
)
  return FletcherPenaltyNLP(
    nlp,
    σ_0,
    rho_0,
    delta_0,
    linear_system_solver,
    hessian_approx;
    x0 = nlp.meta.x0,
  )
end

function obj(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T <: AbstractFloat}
  @lencheck nlp.meta.nvar x
  f = obj(nlp.nlp, x)
  nlp.fx = f
  c = cons(nlp.nlp, x)
  nlp.cx = c
  g = grad(nlp.nlp, x)
  nlp.gx = g
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  rhs1 = vcat(g, T(σ) * c)

  _sol1, _ = nlp.linear_system_solver(nlp, x, rhs1, nothing)

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon
  gs, ys = _sol1[1:nvar], _sol1[(nvar + 1):(nvar + ncon)]
  nlp.ys = ys
  fx = f - dot(c, ys) + T(ρ) / 2 * dot(c, c)

  return fx
end

function grad!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  gx::AbstractVector{T},
) where {T <: AbstractFloat}
  @lencheck nlp.meta.nvar x gx
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  c = cons(nlp.nlp, x)
  nlp.cx = c
  g = grad(nlp.nlp, x)
  nlp.gx = g
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  rhs1 = vcat(g, T(σ) * c)
  rhs2 = vcat(zeros(T, nvar), c)

  _sol1, _sol2 = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

  gs, ys = _sol1[1:nvar], _sol1[(nvar + 1):(nvar + ncon)]
  nlp.ys = ys
  v, w = _sol2[1:nvar], _sol2[(nvar + 1):(nvar + ncon)]
  Hsv = hprod(nlp.nlp, x, ys, v, obj_weight = one(T))
  Sstw = hprod(nlp.nlp, x, w, gs; obj_weight = zero(T))
  Ysc = Hsv - T(σ) * v - Sstw

  #regularization term
  if ρ > 0.0
    Jc = jtprod(nlp.nlp, x, c * T(ρ))
    gx .= gs - Ysc + Jc
  else
    gx .= gs - Ysc
  end

  return gx
end

function objgrad!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  gx::AbstractVector{T},
) where {T <: AbstractFloat}
  @lencheck nlp.meta.nvar x gx
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  f = obj(nlp.nlp, x)
  nlp.fx = f
  c = cons(nlp.nlp, x)
  nlp.cx = c
  g = grad(nlp.nlp, x)
  nlp.gx = g
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  rhs1 = vcat(g, T(σ) * c)
  rhs2 = vcat(zeros(T, nvar), c)

  _sol1, _sol2 = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

  gs, ys = _sol1[1:nvar], _sol1[(nvar + 1):(nvar + ncon)]
  nlp.ys = ys
  v, w = _sol2[1:nvar], _sol2[(nvar + 1):(nvar + ncon)]
  Hsv = hprod(nlp.nlp, x, ys, v, obj_weight = one(T))
  Sstw = hprod(nlp.nlp, x, w, gs; obj_weight = zero(T))
  Ysc = Hsv - T(σ) * v - Sstw

  Jc = jtprod(nlp.nlp, x, c * T(ρ))

  #regularization term
  if ρ > 0.0
    Jc = jtprod(nlp.nlp, x, c * T(ρ))
    fx = f - dot(c, ys) + T(ρ) / 2 * dot(c, c)
    gx .= gs - Ysc + Jc
  else
    fx = f - dot(c, ys)
    gx .= gs - Ysc
  end

  return fx, gx
end

function hess_structure!(
  nlp::FletcherPenaltyNLP,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  f = obj(nlp.nlp, x)
  nlp.fx = f
  c = cons(nlp.nlp, x)
  nlp.cx = c
  g = grad(nlp.nlp, x)
  nlp.gx = g
  A = jac(nlp.nlp, x)
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  rhs1 = vcat(g, T(σ) * c)
  rhs2 = vcat(zeros(T, nvar), c)

  _sol1, _sol2 = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

  gs, ys = _sol1[1:nvar], _sol1[(nvar + 1):(nvar + ncon)]

  Hs = Symmetric(hess(nlp.nlp, x, -ys), :L)
  In = Matrix(I, nvar, nvar)
  Im = Matrix(I, ncon, ncon)
  τ = T(max(nlp.δ, 1e-14))
  invAtA = pinv(Matrix(A * A') + τ * Im) #inv(Matrix(A*A') + τ * Im) #Euh... wait !

  AinvAtA = A' * invAtA
  Pt = AinvAtA * A

  #regularization term
  if ρ > 0.0
    J = jac(nlp.nlp, x)
    Hc = hess(nlp.nlp, x, c * T(ρ), obj_weight = zero(T))
    Hcrho = Hc + T(ρ) * J' * J
    Hx = (In - Pt) * Hs - Hs * Pt + 2 * T(σ) * Pt + Hcrho
  else
    Hx = (In - Pt) * Hs - Hs * Pt + 2 * T(σ) * Pt
  end

  if nlp.hessian_approx == Val(1)
    Ss = Array{T, 2}(undef, ncon, nvar)
    for j = 1:ncon
      Ss[j, :] = gs' * Symmetric(jth_hess(nlp.nlp, x, j), :L)
    end
    Hx += -AinvAtA * Ss - Ss' * invAtA * A
  end

  k = 1
  for j = 1:nvar
    for i = j:nvar
      vals[k] = obj_weight * Hx[i, j]
      k += 1
    end
  end

  return vals
end

function hess_coord!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  #This is an unconstrained optimization problem
  return hess_coord!(nlp, x, vals; obj_weight = obj_weight)
end

function hprod!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  return hprod!(nlp, x, v, Hv, obj_weight = obj_weight)
end

function hprod!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  f = obj(nlp.nlp, x)
  nlp.fx = f
  c = cons(nlp.nlp, x)
  nlp.cx = c
  g = grad(nlp.nlp, x)
  nlp.gx = g
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  τ = T(max(δ, 1e-14))

  rhs1 = vcat(g, T(σ) * c)
  rhs2 = vcat(zeros(T, nvar), c)

  _sol1, _sol2 = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

  gs, ys = _sol1[1:nvar], _sol1[(nvar + 1):(nvar + ncon)]

  Hsv = hprod(nlp.nlp, x, -ys, v, obj_weight = one(T))
  #Hsv    = hprod(nlp.nlp, x, -ys+ρ*c, v, obj_weight = 1.0)

  pt_rhs1 = vcat(v, zeros(T, ncon))
  pt_rhs2 = vcat(Hsv, zeros(T, ncon))
  pt_sol1, pt_sol2 = nlp.linear_system_solver(nlp, x, pt_rhs1, pt_rhs2)
  Ptv = v - pt_sol1[1:nvar]
  PtHsv = Hsv - pt_sol2[1:nvar]
  HsPtv = hprod(nlp.nlp, x, -ys, Ptv, obj_weight = one(T))

  if nlp.hessian_approx == Val(2) && ρ > 0.0
    Jv = jprod(nlp.nlp, x, v)
    JtJv = jtprod(nlp.nlp, x, Jv)
    Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    Hv .= Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv + T(ρ) * (Hcv + JtJv)
  elseif nlp.hessian_approx == Val(2)
    Hv .= Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv
  elseif nlp.hessian_approx == Val(1) && ρ > 0.0
    Jv = jprod(nlp.nlp, x, v)
    JtJv = jtprod(nlp.nlp, x, Jv)
    Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    Jt = jac_op(nlp.nlp, x)'
    invJtJJv = cgls(Jt, v, λ = τ)[1] #invAtA * Jv #cgls(JtJ, Jv)[1]
    SsinvJtJJv = hprod(nlp.nlp, x, invJtJJv, gs, obj_weight = zero(T))

    Ssv = ghjvprod(nlp.nlp, x, gs, v)
    JtJ = jac_op(nlp.nlp, x) * jac_op(nlp.nlp, x)'
    #### TEMP ##########################
    #A = jac(nlp.nlp, x)
    #Im = Matrix(I, ncon, ncon)
    #invAtA = inv(Matrix(A*A') + τ * Im)
    #invJtJSs = invAtA * Ssv
    ###################################
    (invJtJSsv, stats) = minres(JtJ, Ssv, λ = τ) #fix after Krylov.jl #256
    JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)

    Hv .= Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv + T(ρ) * (Hcv + JtJv) - JtinvJtJSsv - SsinvJtJJv
  elseif nlp.hessian_approx == Val(1)
    Jv = jprod(nlp.nlp, x, v)
    Jt = jac_op(nlp.nlp, x)'
    invJtJJv = cgls(Jt, v, λ = τ)[1]
    SsinvJtJJv = hprod(nlp.nlp, x, invJtJJv, gs, obj_weight = zero(T))

    Ssv = ghjvprod(nlp.nlp, x, gs, v)
    JtJ = jac_op(nlp.nlp, x) * jac_op(nlp.nlp, x)'
    #### TEMP ##########################
    #A = jac(nlp.nlp, x)
    #Im = Matrix(I, ncon, ncon)
    #invAtA = inv(Matrix(A*A') + τ * Im)
    #invJtJSs = invAtA * Ssv
    ###################################
    (invJtJSsv, stats) = minres(JtJ, Ssv, λ = τ) #fix after Krylov.jl #256
    #@show norm(invJtJSs - invJtJSsv), norm(Ssv - JtJ * invJtJSsv - τ * invJtJSsv)
    JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)

    Hv .= Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv - JtinvJtJSsv - SsinvJtJJv
  end

  Hv .*= obj_weight
  return Hv
end
