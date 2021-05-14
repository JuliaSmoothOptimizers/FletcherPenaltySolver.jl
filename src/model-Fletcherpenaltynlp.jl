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

include("solve_two_systems_struct.jl")

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

Example:
fp_sos  = FletcherPenaltyNLP(nlp, 0.1, _solve_with_linear_operator)
"""
mutable struct FletcherPenaltyNLP{
  S <: AbstractFloat,
  T <: AbstractVector{S},
  A <: Union{Val{1}, Val{2}},
  P <: Real,
  QDS <: QDSolver,
} <: AbstractNLPModel
  meta::AbstractNLPModelMeta
  counters::Counters
  nlp::AbstractNLPModel

  # Evaluation of the FletcherPenaltyNLP functions contains info on nlp:
  fx::S
  cx::T
  gx::T
  ys::T
  gs::T
  xk::T # last iterate

  # Pre-allocated space:
  _sol1::T
  _sol2::T
  Hsv::T 
  Sstw::T
  Jcρ::T

  # Problem parameter
  σ::P
  ρ::P
  δ::P
  η::P

  qdsolver::QDS
  linear_system_solver::Function # to be removed

  hessian_approx::A
end

function FletcherPenaltyNLP(nlp, σ, linear_system_solver, hessian_approx; x0 = nlp.meta.x0)
  S = eltype(x0)
  nvar = nlp.meta.nvar

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
  )

  return FletcherPenaltyNLP(
    meta,
    Counters(),
    nlp,
    S(NaN),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    σ,
    zero(typeof(σ)),
    zero(typeof(σ)),
    zero(typeof(σ)),
    IterativeSolver(nlp.meta.ncon, nlp.meta.nvar, S(NaN)),
    linear_system_solver,
    hessian_approx,
  )
end

function FletcherPenaltyNLP(nlp, σ, ρ, δ, linear_system_solver, hessian_approx; x0 = nlp.meta.x0)
  S = eltype(x0)
  nvar = nlp.meta.nvar

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
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
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    σ,
    ρ,
    δ,
    zero(typeof(σ)),
    IterativeSolver(nlp.meta.ncon, nlp.meta.nvar, S(NaN)),
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

@memoize function main_obj(nlp::FletcherPenaltyNLP, x::AbstractVector)
  fx = obj(nlp.nlp, x)
  return fx
end

@memoize function main_grad(nlp::FletcherPenaltyNLP, x::AbstractVector)
  return grad(nlp.nlp, x)
end

@memoize function main_cons(nlp::FletcherPenaltyNLP, x::AbstractVector)
  return cons(nlp.nlp, x)
end

@memoize function main_jac(nlp::FletcherPenaltyNLP, x::AbstractVector)
  return jac(nlp.nlp, x)
end

# no need to memoize as it is only used in obj
function linear_system1(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  g = nlp.gx
  c = nlp.cx
  σ = nlp.σ
  rhs1 = vcat(g, T(σ) * c)

  _sol1 = nlp.linear_system_solver(nlp, x, rhs1, nothing)
  #nlp._sol1 .= _sol1
  return _sol1
end

@memoize function linear_system2(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  g = nlp.gx
  c = nlp.cx
  σ = nlp.σ
  rhs1 = vcat(g, T(σ) * c)
  rhs2 = vcat(zeros(T, nlp.meta.nvar), c)

  _sol1, _sol2 = nlp.linear_system_solver(nlp, x, rhs1, rhs2)
  # nlp._sol1 .= _sol1
  # nlp._sol2 .= _sol2

  return _sol1, _sol2
end

function obj(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T <: AbstractFloat}
  nvar = nlp.meta.nvar
  @lencheck nvar x
  increment!(nlp, :neval_obj)
  nlp.fx = main_obj(nlp, x)
  f = nlp.fx
  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx

  _sol1 = linear_system1(nlp, x)
  nlp.ys .= _sol1[(nvar + 1):(nvar + nlp.nlp.meta.ncon)]

  fx = f - dot(c, nlp.ys) + T(nlp.ρ) / 2 * dot(c, c)
  if nlp.η > 0.0
    fx .+= T(nlp.η) / 2 * norm(x - nlp.xk)^2
  end

  return fx
end

function grad!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  gx::AbstractVector{T},
) where {T <: AbstractFloat}
  @lencheck nlp.meta.nvar x gx
  increment!(nlp, :neval_grad)
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]
  ys = nlp.ys

  v, w = _sol2[1:nvar], _sol2[(nvar + 1):(nvar + ncon)]
  hprod!(nlp.nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod!(nlp.nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  gx .= gs - nlp.Hsv + T(σ) * v + nlp.Sstw

  #regularization term
  if ρ > 0.0
    jtprod!(nlp.nlp, x, c * T(ρ), nlp.Jcρ)
    gx .+= nlp.Jcρ
  end
  if nlp.η > 0.0
    gx .+= nlp.η * (x - nlp.xk)
  end

  return gx
end

function objgrad!(
  nlp::FletcherPenaltyNLP,
  x::AbstractVector{T},
  gx::AbstractVector{T},
) where {T <: AbstractFloat}
  @lencheck nlp.meta.nvar x gx
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  nlp.fx = main_obj(nlp, x); f = nlp.fx
  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]
  ys = nlp.ys

  v, w = _sol2[1:nvar], _sol2[(nvar + 1):(nvar + ncon)]
  hprod!(nlp.nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod!(nlp.nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  gx .= gs - nlp.Hsv + T(σ) * v + nlp.Sstw
  fx = f - dot(c, ys)

  #regularization term
  if ρ > 0.0
    jtprod!(nlp.nlp, x, c * T(ρ), nlp.Jcρ)
    gx .+= nlp.Jcρ # gs - Ysc + Jc
    fx += T(ρ) / 2 * dot(c, c)
  end
  if nlp.η > 0.0
    fx .+= T(nlp.η) / 2 * norm(x - nlp.xk)^2
    gx .+= nlp.η * (x - nlp.xk)
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

  nlp.fx = main_obj(nlp, x)
  f = nlp.fx
  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx
  A = main_jac(nlp, x)

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]
  ys = nlp.ys

  Hs = Symmetric(hess(nlp.nlp, x, -ys), :L)
  In = Matrix(I, nvar, nvar)
  Im = Matrix(I, ncon, ncon)
  τ = T(max(nlp.δ, 1e-14))
  invAtA = pinv(Matrix(A * A') + τ * Im) #inv(Matrix(A*A') + τ * Im) #Euh... wait !

  AinvAtA = A' * invAtA
  Pt = AinvAtA * A

  #regularization term
  if ρ > 0.0
    #J = jac(nlp.nlp, x)
    Hc = hess(nlp.nlp, x, c * T(ρ), obj_weight = zero(T))
    Hcrho = Hc + T(ρ) * A' * A
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

#################################################################"
# REMOVE
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
# END REMOVE
#################################################################"

function hprod!(
  nlp::FletcherPenaltyNLP{S, Tt, Val{2}, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T, S, Tt, P, QDS}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  τ = T(max(δ, 1e-14)) # should be a parameter in the solver structure

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  nlp.fx = main_obj(nlp, x)
  f = nlp.fx
  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]
  ys = nlp.ys

  hprod!(nlp.nlp, x, -ys, v, nlp.Hsv, obj_weight = one(T))
  #Hsv    = hprod(nlp.nlp, x, -ys+ρ*c, v, obj_weight = 1.0)

  pt_rhs1 = vcat(v, zeros(T, ncon))
  pt_rhs2 = vcat(nlp.Hsv, zeros(T, ncon))
  pt_sol1, pt_sol2 = nlp.linear_system_solver(nlp, x, pt_rhs1, pt_rhs2)
  Ptv = v - pt_sol1[1:nvar]
  PtHsv = nlp.Hsv - pt_sol2[1:nvar]
  HsPtv = hprod(nlp.nlp, x, -ys, Ptv, obj_weight = one(T))

  Hv .= nlp.Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv

  if ρ > 0.0
    Jv = jprod(nlp.nlp, x, v)
    JtJv = jtprod(nlp.nlp, x, Jv)
    Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    Hv .+= T(ρ) * (Hcv + JtJv)
  end

  Hv .*= obj_weight
  return Hv
end

function hprod!(
  nlp::FletcherPenaltyNLP{S, Tt, Val{1}, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T, S, Tt, P, QDS}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  τ = T(max(δ, 1e-14)) # should be a parameter in the solver structure

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  nlp.fx = main_obj(nlp, x)
  f = nlp.fx
  nlp.gx .= main_grad(nlp, x)
  g = nlp.gx
  nlp.cx .= main_cons(nlp, x)
  c = nlp.cx

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]
  ys = nlp.ys

  hprod!(nlp.nlp, x, -ys, v, nlp.Hsv, obj_weight = one(T))

  pt_rhs1 = vcat(v, zeros(T, ncon))
  pt_rhs2 = vcat(nlp.Hsv, zeros(T, ncon))
  pt_sol1, pt_sol2 = nlp.linear_system_solver(nlp, x, pt_rhs1, pt_rhs2)
  Ptv = v - pt_sol1[1:nvar]
  PtHsv = nlp.Hsv - pt_sol2[1:nvar]
  HsPtv = hprod(nlp.nlp, x, -ys, Ptv, obj_weight = one(T))

  Jv = jprod(nlp.nlp, x, v)
  Jt = jac_op(nlp.nlp, x)'
  invJtJJv = cgls(Jt, v, λ = τ)[1]
  SsinvJtJJv = hprod(nlp.nlp, x, invJtJJv, gs, obj_weight = zero(T))

  Ssv = ghjvprod(nlp.nlp, x, gs, v)
  JtJ = jac_op(nlp.nlp, x) * jac_op(nlp.nlp, x)'
  (invJtJSsv, stats) = minres(JtJ, Ssv, λ = τ) #fix after Krylov.jl #256
  JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)

  Hv .= nlp.Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv - JtinvJtJSsv - SsinvJtJJv

  if ρ > 0.0
    Jv = jprod(nlp.nlp, x, v)
    JtJv = jtprod(nlp.nlp, x, Jv)
    Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    Hv .+= T(ρ) * (Hcv + JtJv)
  end

  Hv .*= obj_weight
  return Hv
end

# gs, ys, v, w = _compute_ys_gs!(nlp, x)
function _compute_ys_gs!(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  nlp.fx = main_obj(nlp, x)
  nlp.gx .= main_grad(nlp, x)
  nlp.cx .= main_cons(nlp, x)

  _sol1, _sol2 = linear_system2(nlp, x)

  gs = _sol1[1:nvar]
  nlp.ys .= _sol1[(nvar + 1):(nvar + ncon)]

  v, w = _sol2[1:nvar], _sol2[(nvar + 1):(nvar + ncon)]

  return gs, nlp.ys, v, w
end
