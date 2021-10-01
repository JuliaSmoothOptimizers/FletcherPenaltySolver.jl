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
`FletcherPenaltyNLP(:: AbstractNLPModel; σ_0 :: Real = 1.0)`

Notes:
- Evaluation of the obj, grad, objgrad functions evaluate functions from the orginial nlp.
These values are stored in `fx`, `cx`, `gx`.
- The value of the penalty vector `ys` is also stored.

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
} <: AbstractNLPModel{S, T}
  meta::AbstractNLPModelMeta{S, T}
  counters::Counters
  nlp::AbstractNLPModel{S, T}

  # Evaluation of the FletcherPenaltyNLP functions contains info on nlp:
  shahx::UInt64 # the x at which fx, cx, gx, ys, and gs are computed
  fx::S
  cx::T
  gx::T
  Aop::LinearOperators.LinearOperator{S}
  ys::T
  gs::T
  xk::T # last iterate

  # Pre-allocated space:
  v::T
  w::T #2nd linear system
  _sol1::T
  _sol2::T
  Hsv::T
  Sstw::T
  Jcρ::T
  Jv::T
  Ss::Array{S, 2} # only when Val(1)

  # Problem parameter
  σ::P
  ρ::P
  δ::P
  η::P

  qdsolver::QDS

  hessian_approx::A
  explicit_linear_constraints::Bool
  ncon_pen::Int # number of penalized constraints
  lin::Array{Int, 1} # indices of non-explicitly maintained constraints
end

function FletcherPenaltyNLP(nlp, σ, hessian_approx; kwargs...)
  return FletcherPenaltyNLP(nlp, σ, hessian_approx, nlp.meta.x0; kwargs...)
end

function FletcherPenaltyNLP(
  nlp,
  σ,
  hessian_approx,
  x0::AbstractVector{S};
  explicit_linear_constraints = true,
  qds = LDLtSolver(nlp, S(0)),
) where {S}
  nvar = nlp.meta.nvar
  if explicit_linear_constraints
    (rows, cols) = jac_structure(nlp)
    nnzj = length(findall(i -> rows[i] ∈ nlp.meta.lin, 1:nlp.meta.nnzj))
  else
    nnzj = 0
  end
  meta = NLPModelMeta{S, Vector{S}}(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    ncon = explicit_linear_constraints ? nlp.meta.nlin : 0,
    lcon = explicit_linear_constraints ? nlp.meta.lcon[nlp.meta.lin] : zeros(S, 0),
    ucon = explicit_linear_constraints ? nlp.meta.ucon[nlp.meta.lin] : zeros(S, 0),
    nnzj = nnzj,
    nlin = explicit_linear_constraints ? nlp.meta.nlin : 0,
    lin  = explicit_linear_constraints ? nlp.meta.lin : Int[],
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
  )
  ncon_pen = explicit_linear_constraints ? nlp.meta.ncon - nlp.meta.nlin : nlp.meta.ncon
  lin = explicit_linear_constraints ? setdiff(1:nlp.meta.ncon, nlp.meta.lin) : Int[]

  return FletcherPenaltyNLP(
    meta,
    Counters(),
    nlp,
    zero(UInt64),
    S(NaN),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    LinearOperator{S}(nlp.meta.ncon, nlp.meta.nvar, false, false, v -> v, v -> v, v -> v),
    Vector{S}(undef, nlp.meta.ncon), # ncon_pen
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Array{S, 2}(undef, nlp.meta.ncon, nlp.meta.nvar),
    σ,
    zero(typeof(σ)),
    zero(typeof(σ)),
    zero(typeof(σ)),
    qds,
    hessian_approx,
    explicit_linear_constraints,
    ncon_pen,
    lin,
  )
end

function FletcherPenaltyNLP(nlp, σ, ρ, δ, hessian_approx; kwargs...)
  return FletcherPenaltyNLP(nlp, σ, ρ, δ, hessian_approx, nlp.meta.x0; kwargs...)
end

function FletcherPenaltyNLP(
  nlp,
  σ,
  ρ,
  δ,
  hessian_approx,
  x0::AbstractVector{S};
  explicit_linear_constraints = true,
  qds = LDLtSolver(nlp, S(0)), #IterativeSolver(nlp, S(NaN)),
) where {S}
  nvar = nlp.meta.nvar
  if explicit_linear_constraints
    (rows, cols) = jac_structure(nlp)
    nnzj = length(findall(i -> rows[i] ∈ nlp.meta.lin, 1:nlp.meta.nnzj))
  else
    nnzj = 0
  end
  meta = NLPModelMeta{S, Vector{S}}(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    ncon = explicit_linear_constraints ? nlp.meta.nlin : 0,
    lcon = explicit_linear_constraints ? nlp.meta.lcon[nlp.meta.lin] : zeros(S, 0),
    ucon = explicit_linear_constraints ? nlp.meta.ucon[nlp.meta.lin] : zeros(S, 0),
    nnzj = nnzj,
    nlin = explicit_linear_constraints ? nlp.meta.nlin : 0,
    lin  = explicit_linear_constraints ? nlp.meta.lin : Int[],
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
  )
  counters = Counters()
  ncon_pen = explicit_linear_constraints ? nlp.meta.ncon - nlp.meta.nlin : nlp.meta.ncon
  lin = explicit_linear_constraints ? setdiff(1:nlp.meta.ncon, nlp.meta.lin) : 1:nlp.meta.ncon
  return FletcherPenaltyNLP(
    meta,
    counters,
    nlp,
    zero(UInt64),
    S(NaN),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    LinearOperator{S}(nlp.meta.ncon, nlp.meta.nvar, false, false, v -> v, v -> v, v -> v),
    Vector{S}(undef, nlp.meta.ncon), # ncon_pen
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar + nlp.meta.ncon),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.ncon),
    Array{S, 2}(undef, nlp.meta.ncon, nlp.meta.nvar),
    σ,
    ρ,
    δ,
    zero(typeof(σ)),
    qds,
    hessian_approx,
    explicit_linear_constraints,
    ncon_pen,
    lin,
  )
end

#Set of functions solving two linear systems with different rhs.
# solve_two_extras, solve_two_least_squares, solve_two_mixed
include("solve_linear_system.jl")

function FletcherPenaltyNLP(
  nlp::AbstractNLPModel;
  σ_0::Real = one(eltype(nlp.meta.x0)),
  ρ_0::Real = zero(eltype(nlp.meta.x0)),
  δ_0::Real = zero(eltype(nlp.meta.x0)),
  hessian_approx = Val(2),
  x0 = nlp.meta.x0,
  kwargs...,
)
  return FletcherPenaltyNLP(nlp, σ_0, ρ_0, δ_0, hessian_approx, x0; kwargs...)
end

function linear_system2(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon
  g = nlp.gx
  c = nlp.cx
  σ = nlp.σ
  #rhs1 = vcat(g, T(σ) * c)
  #rhs2 = vcat(zeros(T, nlp.meta.nvar), c)

  (p1, q1, p2, q2) = solve_two_mixed(nlp, x, g, c)
  # nlp._sol1 .= _sol1
  # nlp._sol2 .= _sol2

  return p1, q1, p2, q2
end

# gs, ys, v, w = _compute_ys_gs!(nlp, x)
# Improve allocs and all here !
function _compute_ys_gs!(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon
  shahx = hash(x)
  if shahx != nlp.shahx
    nlp.shahx = shahx
    nlp.fx = obj(nlp.nlp, x)
    grad!(nlp.nlp, x, nlp.gx)
    cons!(nlp.nlp, x, nlp.cx)

    p1, q1, p2, q2 = linear_system2(nlp, x)

    nlp.gs .= p1 + T(nlp.σ) * p2 #_sol1[1:nvar]
    nlp.ys .= q1 + T(nlp.σ) * q2 #_sol1[(nvar + 1):(nvar + ncon)]

    nlp.v .= p2 #_sol2[1:nvar]
    nlp.w .= q2 #_sol2[(nvar + 1):(nvar + ncon)]
  end

  return nlp.gs, nlp.ys, nlp.v, nlp.w
end

function obj(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T <: AbstractFloat}
  nvar = nlp.meta.nvar
  @lencheck nvar x
  increment!(nlp, :neval_obj)

  #_sol1 = linear_system1(nlp, x)
  #nlp.ys .= _sol1[(nvar + 1):(nvar + nlp.nlp.meta.ncon)]
  _, ys, _, _ = _compute_ys_gs!(nlp, x)

  f = nlp.fx
  c = nlp.cx

  fx = f - dot(c, nlp.ys) + T(nlp.ρ) / 2 * dot(c, c)
  if nlp.η > 0.0
    fx += T(nlp.η) / 2 * norm(x - nlp.xk)^2
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

  gs, ys, v, w = _compute_ys_gs!(nlp, x)
  g = nlp.gx
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  hprod!(nlp.nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod!(nlp.nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  @. gx = gs - nlp.Hsv + T(σ) * v + nlp.Sstw

  #regularization term
  if ρ > 0.0
    jtprod!(nlp.nlp, x, c, nlp.Jcρ) # J' * c * ρ
    @. gx += nlp.Jcρ * T(ρ) # Should the product by rho be done here?
  end
  if nlp.η > 0.0
    @. gx += T(nlp.η) * (x - nlp.xk)
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

  gs, ys, v, w = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  hprod!(nlp.nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod!(nlp.nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  @. gx = gs - nlp.Hsv + T(σ) * v + nlp.Sstw
  fx = f - dot(c, ys)

  #regularization term
  if ρ > 0.0
    jtprod!(nlp.nlp, x, c, nlp.Jcρ)
    @. gx += nlp.Jcρ * T(ρ) # gs - Ysc + Jc
    fx += T(ρ) / 2 * dot(c, c)
  end
  if nlp.η > 0.0
    fx += T(nlp.η) / 2 * norm(x - nlp.xk)^2
    @. gx += nlp.η * (x - nlp.xk)
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

  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx
  A = jac(nlp.nlp, x) # If used, this should be allocated probably
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  Hs = Symmetric(hess(nlp.nlp, x, -ys), :L)
  Im = Matrix(I, ncon, ncon)
  τ = T(max(nlp.δ, 1e-14))
  invAtA = pinv(Matrix(A * A') + τ * Im) #inv(Matrix(A*A') + τ * Im) # Euh... wait !
  AinvAtA = A' * invAtA
  Pt = AinvAtA * A

  Hx = Hs - Pt * Hs - Hs * Pt + 2 * T(σ) * Pt
  #regularization term
  if ρ > 0.0
    Hc = hess(nlp.nlp, x, c * T(ρ), obj_weight = zero(T))
    Hx += Hc + T(ρ) * A' * A
  end

  if nlp.hessian_approx == Val(1)
    for j = 1:ncon
      nlp.Ss[j, :] = gs' * Symmetric(jth_hess(nlp.nlp, x, j), :L)
    end
    Hx += -AinvAtA * nlp.Ss - nlp.Ss' * invAtA * A
  end

  #=
  if nlp.η > 0.0
    In = Matrix(I, nvar, nvar)
    Hx += T(nlp.η) * In
  end
  =#

  k = 1
  for j = 1:nvar
    for i = j:nvar
      vals[k] = obj_weight * Hx[i, j]
      if i == j && nlp.η > 0.0
        vals[k] += obj_weight * T(nlp.η)
      end
      k += 1
    end
  end

  return vals
end

function hprod!(
  nlp::FletcherPenaltyNLP{S, Tt, Val{2}, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T, S, Tt, P, QDS}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx

  @. nlp.Jv = -ys
  hprod!(nlp.nlp, x, nlp.Jv, v, nlp.Hsv, obj_weight = one(T))
  #Hsv    = hprod(nlp.nlp, x, -ys+ρ*c, v, obj_weight = 1.0)

  (p1, _, p2, _) = solve_two_least_squares(nlp, x, v, nlp.Hsv)
  @. nlp.Hsv = v - p1 # Ptv = v - p1
  hprod!(nlp.nlp, x, nlp.Jv, nlp.Hsv, nlp.v, obj_weight = one(T)) # HsPtv = hprod(nlp.nlp, x, -ys, Ptv, obj_weight = one(T))

  # PtHsv = nlp.Hsv - pt_sol2[1:nvar]
  # Hv .= nlp.Hsv - PtHsv - nlp.v + 2 * T(σ) * Ptv
  # Hv .= pt_sol2[1:nvar] - nlp.v + 2 * T(σ) * Ptv
  @. Hv = p2 - nlp.v + 2 * T(σ) * nlp.Hsv

  if ρ > 0.0
    jprod!(nlp.nlp, x, v, nlp.Jv)
    jtprod!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    hprod!(nlp.nlp, x, c, v, nlp.v, obj_weight = zero(T)) # Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    @. Hv += nlp.v + T(ρ) * nlp.Jcρ
  end
  if nlp.η > 0.0
    @. Hv += T(nlp.η) * v
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

  nvar = nlp.meta.nvar
  ncon = nlp.nlp.meta.ncon

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx

  hprod!(nlp.nlp, x, -ys, v, nlp.Hsv, obj_weight = one(T))

  (p1, _, p2, _) = solve_two_least_squares(nlp, x, v, nlp.Hsv)
  @. nlp.Hsv = v - p1 # Ptv = v - p1
  # PtHsv = nlp.Hsv - p2

  @. nlp.Jv = -ys
  # HsPtv = nlp.Jcρ
  hprod!(nlp.nlp, x, nlp.Jv, nlp.Hsv, nlp.Jcρ, obj_weight = one(T))

  ghjvprod!(nlp.nlp, x, gs, v, nlp.w) # Ssv = ghjvprod(nlp.nlp, x, gs, v)
  Ssv = nlp.w
  invJtJJv, invJtJSsv = solve_two_extras(nlp, x, v, Ssv)
  jtprod!(nlp.nlp, x, invJtJSsv, nlp.v) # JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)

  # @. Hv = nlp.Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv - nlp.v - SsinvJtJJv
  # @. Hv = p2 - HsPtv + 2 * T(σ) * nlp.Hsv - nlp.v - SsinvJtJJv
  # @. Hv = p2 - nlp.Jcρ + 2 * T(σ) * nlp.Hsv - nlp.v - SsinvJtJJv
  @. Hv = p2 - nlp.Jcρ + 2 * T(σ) * nlp.Hsv - nlp.v
  hprod!(nlp.nlp, x, invJtJJv, gs, nlp.Jcρ, obj_weight = zero(T)) # SsinvJtJJv
  @. Hv -= nlp.Jcρ

  if ρ > 0.0
    jprod!(nlp.nlp, x, v, nlp.Jv)
    jtprod!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    hprod!(nlp.nlp, x, c, v, nlp.v, obj_weight = zero(T)) # Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))
    @. Hv += T(ρ) * (nlp.v + nlp.Jcρ)
  end
  if nlp.η > 0.0
    @. Hv += T(nlp.η) * v
  end

  Hv .*= obj_weight
  return Hv
end

function NLPModels.cons!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  c::AbstractVector{T},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  nlp.cx = cons!(nlp.nlp, x, nlp.cx)
  c .= nlp.cx[nlp.nlp.meta.lin]
  return c
end

function NLPModels.jac_coord!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  vals::AbstractVector{T},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  if nlp.explicit_linear_constraints
    valjac = jac_coord(nlp.nlp, x)
    (rows, cols) = jac_structure(nlp.nlp)
    vals .= valjac[findall(i -> rows[i] ∈ nlp.nlp.meta.lin, 1:nlp.nlp.meta.nnzj)]
  end
  return vals
end

function NLPModels.jac_structure!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nnzj rows cols
  if nlp.explicit_linear_constraints
    (jrows, jcols) = jac_structure(nlp.nlp)
    rows .= jrows[findall(i -> jrows[i] ∈ nlp.nlp.meta.lin, 1:nlp.nlp.meta.nnzj)]
    cols .= jcols[findall(i -> jrows[i] ∈ nlp.nlp.meta.lin, 1:nlp.nlp.meta.nnzj)]
  end
  return (rows, cols)
end

function NLPModels.jprod!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  Jv .= jprod(nlp.nlp, x, v)[nlp.nlp.meta.lin]
  return Jv
end

function NLPModels.jtprod!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Jtv::AbstractVector,
  ) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  vlin = zeros(T, nlp.nlp.meta.ncon)
  vlin[nlp.nlp.meta.lin] = v
  Jtv .= jtprod(nlp.nlp, x, vlin)
  return Jtv
end

function NLPModels.hess(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  y::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  return hess(nlp, x, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  return hess_coord!(nlp, x, vals, obj_weight = obj_weight)
end

function NLPModels.hprod!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon y
  return hprod!(nlp, x, v, Hv, obj_weight = obj_weight)
end
