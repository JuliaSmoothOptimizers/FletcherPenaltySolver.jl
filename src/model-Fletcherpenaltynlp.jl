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
    FletcherPenaltyNLP(nlp, σ, hessian_approx; kwargs...)
    FletcherPenaltyNLP(nlp, σ, hessian_approx, x; qds = LDLtSolver(nlp, S(0)))
    FletcherPenaltyNLP(nlp, σ, ρ, δ, hessian_approx; kwargs...)
    FletcherPenaltyNLP(nlp, σ, ρ, δ, hessian_approx, x; qds = LDLtSolver(nlp, S(0)))
    FletcherPenaltyNLP(nlp; σ_0::Real = one(eltype(nlp.meta.x0)), ρ_0::Real = zero(eltype(nlp.meta.x0)), δ_0::Real = zero(eltype(nlp.meta.x0)), hessian_approx = Val(2), x0 = nlp.meta.x0, kwargs...)

We consider here the implementation of Fletcher's exact penalty method for
the minimization problem:

```math
    minₓ    f(x)    s.t.    c(x) = ℓ, l ≤ x ≤ u
```

using Fletcher penalty function:
```math   
    minₓ    f(x) - (c(x) - ℓ)^T ys(x) + 0.5 ρ ||c(x) - ℓ||²₂    s.t.    l ≤ x ≤ u
```
where
```math
    ys(x)    ∈    arg min\\_y    0.5 ||A(x)y - g(x)||²₂ + σ (c(x) - ℓ)^T y + 0.5 || δ ||²₂
```

# Arguments
- `nlp::AbstractNLPModel`: the model solved, see `NLPModels.jl`;
- `x::AbstractVector`: Initial guess. If `x` is not specified, then `nlp.meta.x0` is used;
- `σ`, `ρ`, `δ` parameters of the subproblem;
- `hessian_approx` either `Val(1)` or `Val(2)` for the hessian approximation.
- `qds`: solver structure for the linear algebra computations, see [`LDLtSolver`](@ref) or [`IterativeSolver`](@ref).

# Notes:
- Evaluation of the `obj`, `grad`, `objgrad` functions evaluate functions from the orginial `nlp`. These values are stored in `fx`, `cx`, `gx`.
- The value of the penalty vector `ys` is also stored.
- The hessian's structure is dense.

# Examples
```julia
julia> using FletcherPenaltyNLPSolver, ADNLPModels
julia> nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
julia> fp_sos  = FletcherPenaltyNLP(nlp)
```
"""
mutable struct FletcherPenaltyNLP{
  S,
  T,
  A <: Union{Val{1}, Val{2}},
  P <: Real,
  QDS <: QDSolver,
  Pb,
} <: AbstractNLPModel{S, T}
  meta::NLPModelMeta{S, T}
  counters::Counters
  nlp::Pb

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
  lag_mul::T # size `ncon` if explicit_linear_constraints

  # Problem parameter
  σ::P
  ρ::P
  δ::P
  η::P

  qdsolver::QDS

  hessian_approx::A
  explicit_linear_constraints::Bool
end

function FletcherPenaltyNLP(nlp, σ, hessian_approx; kwargs...)
  return FletcherPenaltyNLP(nlp, σ, hessian_approx, nlp.meta.x0; kwargs...)
end

function FletcherPenaltyNLP(
  nlp,
  σ,
  hessian_approx,
  x0::AbstractVector{S};
  explicit_linear_constraints = false,
  qds = LDLtSolver(nlp, S(0)),
) where {S}
  nvar = nlp.meta.nvar
  if explicit_linear_constraints
    lin = collect(1:nlp.meta.nlin)
    nnzj = nlp.meta.lin_nnzj
    lin_nnzj = nlp.meta.lin_nnzj
    nln_nnzj = 0
    npen = nlp.meta.nnln
  else
    lin = Int[]
    nnzj = 0
    lin_nnzj = 0
    nln_nnzj = 0
    npen = nlp.meta.ncon
  end
  ncon = explicit_linear_constraints ? nlp.meta.nlin : 0
  meta = NLPModelMeta{S, Vector{S}}(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
    ncon = ncon,
    lcon = explicit_linear_constraints ? nlp.meta.lcon[nlp.meta.lin] : zeros(S, 0),
    ucon = explicit_linear_constraints ? nlp.meta.ucon[nlp.meta.lin] : zeros(S, 0),
    lin = lin,
    nnzj = nnzj,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
  )

  return FletcherPenaltyNLP(
    meta,
    Counters(),
    nlp,
    zero(UInt64),
    S(NaN),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar),
    LinearOperator{S}(npen, nlp.meta.nvar, false, false, v -> v, v -> v, v -> v),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar + npen),
    Vector{S}(undef, nlp.meta.nvar + npen),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, npen),
    Array{S, 2}(undef, npen, nlp.meta.nvar),
    explicit_linear_constraints & (ncon > 0) ? zeros(S, nlp.meta.ncon) : S[], # pre-allocate for hess/hprod
    σ,
    zero(typeof(σ)),
    zero(typeof(σ)),
    zero(typeof(σ)),
    qds,
    hessian_approx,
    explicit_linear_constraints,
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
  explicit_linear_constraints = false,
  qds = LDLtSolver(nlp, S(0)), #IterativeSolver(nlp, S(NaN)),
) where {S}
  nvar = nlp.meta.nvar
  if explicit_linear_constraints
    lin = collect(1:nlp.meta.nlin)
    nnzj = nlp.meta.lin_nnzj
    lin_nnzj = nlp.meta.lin_nnzj
    nln_nnzj = 0
    npen = nlp.meta.nnln
  else
    lin = Int[]
    nnzj = 0
    lin_nnzj = 0
    nln_nnzj = 0
    npen = nlp.meta.ncon
  end
  ncon = explicit_linear_constraints ? nlp.meta.nlin : 0
  meta = NLPModelMeta{S, Vector{S}}(
    nvar,
    x0 = x0,
    nnzh = nvar * (nvar + 1) / 2,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    minimize = true,
    islp = false,
    name = "Fletcher penalization of $(nlp.meta.name)",
    ncon = ncon,
    lcon = explicit_linear_constraints ? nlp.meta.lcon[nlp.meta.lin] : zeros(S, 0),
    ucon = explicit_linear_constraints ? nlp.meta.ucon[nlp.meta.lin] : zeros(S, 0),
    lin = lin,
    nnzj = nnzj,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
  )
  counters = Counters()
  return FletcherPenaltyNLP(
    meta,
    counters,
    nlp,
    zero(UInt64),
    S(NaN),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar),
    LinearOperator{S}(npen, nlp.meta.nvar, false, false, v -> v, v -> v, v -> v),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, npen),
    Vector{S}(undef, nlp.meta.nvar + npen),
    Vector{S}(undef, nlp.meta.nvar + npen),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, nlp.meta.nvar),
    Vector{S}(undef, npen),
    Array{S, 2}(undef, npen, nlp.meta.nvar),
    explicit_linear_constraints & (ncon > 0) ? zeros(S, nlp.meta.ncon) : S[],
    σ,
    ρ,
    δ,
    zero(typeof(σ)),
    qds,
    hessian_approx,
    explicit_linear_constraints,
  )
end

#Set of functions solving two linear systems with different rhs.
# solve_two_extras, solve_two_least_squares, solve_two_mixed
include("solve_linear_system.jl")

function FletcherPenaltyNLP(
  nlp::AbstractNLPModel{T, S};
  σ_0::Real = one(T),
  ρ_0::Real = zero(T),
  δ_0::Real = zero(T),
  hessian_approx = Val(2),
  x0 = nlp.meta.x0,
  kwargs...,
) where {T, S}
  return FletcherPenaltyNLP(nlp, σ_0, ρ_0, δ_0, hessian_approx, x0; kwargs...)
end

"""
    p1, q1, p2, q2 = linear_system2(nlp, x)

Call to `solve_two_mixed(nlp, x, nlp.gx, nlp.cx)`, see [`solve_two_mixed`](@ref).
"""
function linear_system2(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
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

"""
    gs, ys, v, w = _compute_ys_gs!(nlp, x)

Compute the Lagrange multipliers and the gradient of the Lagrangian function in-place.
"""
function _compute_ys_gs!(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T}
  shahx = hash(x)
  if shahx != nlp.shahx
    nlp.shahx = shahx
    nlp.fx = obj(nlp.nlp, x)
    grad!(nlp.nlp, x, nlp.gx)
    cons_norhs!(nlp, x, nlp.cx)

    p1, q1, p2, q2 = linear_system2(nlp, x)

    nlp.gs .= p1 + T(nlp.σ) * p2 #_sol1[1:nvar]
    nlp.ys .= q1 + T(nlp.σ) * q2 #_sol1[(nvar + 1):(nvar + ncon)]

    nlp.v .= p2 #_sol2[1:nvar]
    nlp.w .= q2 #_sol2[(nvar + 1):(nvar + ncon)]
  end

  return nlp.gs, nlp.ys, nlp.v, nlp.w
end

"""
    cons_norhs!(nlp::FletcherPenaltyNLP, x, cx)

Redefine the NLPModel function `cons!` to account for non-zero right-hand side in the equation constraint.
It returns `cons(nlp, x) - nlp.meta.lcon`.
  """
function cons_norhs!(nlp::FletcherPenaltyNLP, x, cx) # evaluation of the origin NLPModel
  if nlp.explicit_linear_constraints & (nlp.meta.ncon > 0) # & (length(cx) == nlp.meta.nnln)
    cons_nln!(nlp.nlp, x, cx)
    cx .-= get_lcon(nlp.nlp)[nlp.nlp.meta.nln]
  elseif nlp.nlp.meta.ncon > 0
    cons!(nlp.nlp, x, cx)
    cx .-= get_lcon(nlp.nlp)
  end
  return cx
end

function cons_norhs!(nlp, x, cx) # evaluation of the origin NLPModel
  if (nlp.meta.ncon > 0) & (length(cx) == nlp.meta.nnln)
    cons_nln!(nlp, x, cx)
    cx .-= get_lcon(nlp)[nlp.meta.nln]
  elseif nlp.meta.ncon > 0
    cons!(nlp, x, cx)
    cx .-= get_lcon(nlp)
  end
  return cx
end

"""
    hprod_nln!(nlp::FletcherPenaltyNLP, x, y, v, Hv; obj_weight = one(S)) where {S}

Redefine the NLPModel function `hprod` to account for Lagrange multiplier of size < ncon.
"""
function hprod_nln!(nlp::FletcherPenaltyNLP, x::AbstractVector{S}, y, v, Hv; obj_weight = one(S)) where {S}
  return if nlp.explicit_linear_constraints & (nlp.meta.ncon > 0)
    nlp.lag_mul .= zero(S)
    nlp.lag_mul[nlp.meta.nln] .= y
    hprod!(nlp.nlp, x, nlp.lag_mul, v, Hv, obj_weight = obj_weight)
  else
    hprod!(nlp.nlp, x, y, v, Hv, obj_weight = obj_weight)
  end
end

"""
    hess_nln_nln!(nlp::FletcherPenaltyNLP, x, y, vals; obj_weight = one(S)) where {S}

Redefine the NLPModel function `hprod` to account for Lagrange multiplier of size < ncon.
"""
function hess_nln_coord!(nlp::FletcherPenaltyNLP, x::AbstractVector{S}, y, vals; obj_weight = one(S)) where {S}
  return if nlp.explicit_linear_constraints & (nlp.meta.ncon > 0)
    nlp.lag_mul .= zero(S)
    nlp.lag_mul[nlp.meta.nln] .= y
    hess_coord!(nlp.nlp, x, nlp.lag_mul, vals, obj_weight = obj_weight)
  else
    hess_coord!(nlp.nlp, x, y, vals, obj_weight = obj_weight)
  end
end

"""
    hess_nln(nlp::FletcherPenaltyNLP, x, y; obj_weight = one(S)) where {S}

Redefine the NLPModel function `hprod` to account for Lagrange multiplier of size < ncon.
"""
function hess_nln(nlp::FletcherPenaltyNLP, x::AbstractVector{S}, y; obj_weight = one(S)) where {S}
  return if nlp.explicit_linear_constraints & (nlp.meta.ncon > 0)
    nlp.lag_mul .= zero(S)
    nlp.lag_mul[nlp.meta.nln] .= y
    hess(nlp.nlp, x, nlp.lag_mul, obj_weight = obj_weight)
  else
    hess(nlp.nlp, x, y, obj_weight = obj_weight)
  end
end

"""
    ghjvprod_nln!(nlp::FletcherPenaltyNLP, x, y, v, Hv; obj_weight = one(S)) where {S}

Redefine the NLPModel function `ghjvprod` to account for Lagrange multiplier of size < ncon.
"""
function ghjvprod_nln!(nlp::FletcherPenaltyNLP, x::AbstractVector{S}, y, v, vals) where {S}
  return if nlp.explicit_linear_constraints & (nlp.meta.ncon > 0)
    ghjvprod!(nlp.nlp, x, y, v, nlp.lag_mul)
    vals .= nlp.lag_mul[nlp.nlp.meta.nln]
  else
    ghjvprod!(nlp.nlp, x, y, v, vals)
  end
end

function obj(nlp::FletcherPenaltyNLP, x::AbstractVector{T}) where {T <: AbstractFloat}
  nvar = get_nvar(nlp)
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
  nvar = get_nvar(nlp)
  @lencheck nvar x gx
  increment!(nlp, :neval_grad)

  gs, ys, v, w = _compute_ys_gs!(nlp, x)
  g = nlp.gx
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  hprod_nln!(nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod_nln!(nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  @. gx = gs - nlp.Hsv + T(σ) * v + nlp.Sstw

  #regularization term
  if ρ > 0.0
    if nlp.explicit_linear_constraints
      jtprod_nln!(nlp.nlp, x, c, nlp.Jcρ) # J' * c * ρ
    else
      jtprod!(nlp.nlp, x, c, nlp.Jcρ) # J' * c * ρ
    end
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
  nvar = get_nvar(nlp)
  @lencheck nvar x gx
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)

  gs, ys, v, w = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  hprod_nln!(nlp, x, ys, v, nlp.Hsv, obj_weight = one(T))
  hprod_nln!(nlp, x, w, gs, nlp.Sstw; obj_weight = zero(T))
  #Ysc = Hsv - T(σ) * v - Sstw
  @. gx = gs - nlp.Hsv + T(σ) * v + nlp.Sstw
  fx = f - dot(c, ys)

  #regularization term
  if ρ > 0.0
    if nlp.explicit_linear_constraints
      jtprod_nln!(nlp.nlp, x, c, nlp.Jcρ) # J' * c * ρ
    else
      jtprod!(nlp.nlp, x, c, nlp.Jcρ) # J' * c * ρ
    end
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
  ncon = nlp.explicit_linear_constraints ? nlp.nlp.meta.nnln : nlp.nlp.meta.ncon
  ind_con = nlp.explicit_linear_constraints ? nlp.nlp.meta.nln : 1:nlp.nlp.meta.ncon

  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx
  if nlp.explicit_linear_constraints
    A = jac_nln(nlp.nlp, x) # If used, this should be allocated probably
  else
    A = jac(nlp.nlp, x) # If used, this should be allocated probably
  end
  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  Hs = Symmetric(hess_nln(nlp, x, -ys), :L)
  Im = Matrix(I, ncon, ncon)
  τ = T(max(nlp.δ, 1e-14))
  invAtA = pinv(Matrix(A * A') + τ * Im) #inv(Matrix(A*A') + τ * Im) # Euh... wait !
  AinvAtA = A' * invAtA
  Pt = AinvAtA * A

  Hx = Hs - Pt * Hs - Hs * Pt + 2 * T(σ) * Pt
  #regularization term
  if ρ > 0.0
    Hc = hess_nln(nlp, x, c * T(ρ), obj_weight = zero(T))
    Hx += Hc + T(ρ) * A' * A
  end

  if nlp.hessian_approx == Val(1)
    k = 0
    for j in ind_con
      k += 1
      nlp.Ss[k, :] = gs' * Symmetric(jth_hess(nlp.nlp, x, j), :L)
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
  nvar = get_nvar(nlp)
  @lencheck nvar x v Hv
  increment!(nlp, :neval_hprod)

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ
  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx

  @. nlp.Jv = -ys
  hprod_nln!(nlp, x, nlp.Jv, v, nlp.Hsv, obj_weight = one(T))
  #Hsv    = hprod(nlp.nlp, x, -ys+ρ*c, v, obj_weight = 1.0)

  (p1, _, p2, _) = solve_two_least_squares(nlp, x, v, nlp.Hsv)
  @. nlp.Hsv = v - p1 # Ptv = v - p1
  hprod_nln!(nlp, x, nlp.Jv, nlp.Hsv, nlp.v, obj_weight = one(T)) # HsPtv = hprod(nlp.nlp, x, -ys, Ptv, obj_weight = one(T))

  # PtHsv = nlp.Hsv - pt_sol2[1:nvar]
  # Hv .= nlp.Hsv - PtHsv - nlp.v + 2 * T(σ) * Ptv
  # Hv .= pt_sol2[1:nvar] - nlp.v + 2 * T(σ) * Ptv
  @. Hv = p2 - nlp.v + 2 * T(σ) * nlp.Hsv

  if ρ > 0.0
    if nlp.explicit_linear_constraints
      jprod_nln!(nlp.nlp, x, v, nlp.Jv)
      jtprod_nln!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    else
      jprod!(nlp.nlp, x, v, nlp.Jv)
      jtprod!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    end
    hprod_nln!(nlp, x, c, v, nlp.v, obj_weight = zero(T)) # Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

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
  nvar = get_nvar(nlp)
  @lencheck nvar x v Hv
  increment!(nlp, :neval_hprod)

  σ, ρ, δ = nlp.σ, nlp.ρ, nlp.δ

  gs, ys, _, _ = _compute_ys_gs!(nlp, x)
  f = nlp.fx
  g = nlp.gx
  c = nlp.cx

  hprod_nln!(nlp, x, -ys, v, nlp.Hsv, obj_weight = one(T))

  (p1, _, p2, _) = solve_two_least_squares(nlp, x, v, nlp.Hsv)
  @. nlp.Hsv = v - p1 # Ptv = v - p1
  # PtHsv = nlp.Hsv - p2

  @. nlp.Jv = -ys
  # HsPtv = nlp.Jcρ
  hprod_nln!(nlp, x, nlp.Jv, nlp.Hsv, nlp.Jcρ, obj_weight = one(T))

  ghjvprod_nln!(nlp, x, gs, v, nlp.w) # Ssv = ghjvprod(nlp.nlp, x, gs, v)
  Ssv = nlp.w
  invJtJJv, invJtJSsv = solve_two_extras(nlp, x, v, Ssv)
  if nlp.explicit_linear_constraints
    jtprod_nln!(nlp.nlp, x, invJtJSsv, nlp.v) # JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)
  else
    jtprod!(nlp.nlp, x, invJtJSsv, nlp.v) # JtinvJtJSsv = jtprod(nlp.nlp, x, invJtJSsv)
  end

  # @. Hv = nlp.Hsv - PtHsv - HsPtv + 2 * T(σ) * Ptv - nlp.v - SsinvJtJJv
  # @. Hv = p2 - HsPtv + 2 * T(σ) * nlp.Hsv - nlp.v - SsinvJtJJv
  # @. Hv = p2 - nlp.Jcρ + 2 * T(σ) * nlp.Hsv - nlp.v - SsinvJtJJv
  @. Hv = p2 - nlp.Jcρ + 2 * T(σ) * nlp.Hsv - nlp.v
  hprod_nln!(nlp, x, invJtJJv, gs, nlp.Jcρ, obj_weight = zero(T)) # SsinvJtJJv
  @. Hv -= nlp.Jcρ

  if ρ > 0.0
    if nlp.explicit_linear_constraints
      jprod_nln!(nlp.nlp, x, v, nlp.Jv)
      jtprod_nln!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    else
      jprod!(nlp.nlp, x, v, nlp.Jv)
      jtprod!(nlp.nlp, x, nlp.Jv, nlp.Jcρ) # JtJv = jtprod(nlp.nlp, x, Jv)
    end
    hprod_nln!(nlp, x, c, v, nlp.v, obj_weight = zero(T)) # Hcv = hprod(nlp.nlp, x, c, v, obj_weight = zero(T))

    @. Hv += T(ρ) * (nlp.v + nlp.Jcρ)
  end
  if nlp.η > 0.0
    @. Hv += T(nlp.η) * v
  end

  Hv .*= obj_weight
  return Hv
end

function NLPModels.cons_lin!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  c::AbstractVector{T},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin c
  increment!(nlp, :neval_cons_lin)
  return cons_lin!(nlp.nlp, x, c)
end

function NLPModels.jac_lin_structure!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.lin_nnzj rows cols
  return jac_lin_structure!(nlp.nlp, rows, cols)
end

function NLPModels.jac_lin_coord!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  vals::AbstractVector{T},
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.lin_nnzj vals
  increment!(nlp, :neval_jac_lin)
  return jac_lin_coord!(nlp.nlp, x, vals)
end

function NLPModels.jprod_lin!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nlin Jv
  increment!(nlp, :neval_jprod_lin)
  return jprod_lin!(nlp.nlp, x, v, Jv)
end

function NLPModels.jtprod_lin!(
  nlp::FletcherPenaltyNLP{S, Tt, V, P, QDS},
  x::AbstractVector{T},
  v::AbstractVector,
  Jtv::AbstractVector,
  ) where {T, S, Tt, V, P, QDS}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nlin v
  increment!(nlp, :neval_jtprod_lin)
  return jtprod_lin!(nlp.nlp, x, v, Jtv)
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
