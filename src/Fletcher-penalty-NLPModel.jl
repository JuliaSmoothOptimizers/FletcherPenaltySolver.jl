import NLPModels: increment!, obj, objgrad, objgrad!, grad!, grad, hess, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!
"""
We consider here the implementation of Fletcher's exact penalty method for
the minimization problem:

min\\_x f(x) s.t. c(x) = 0

using Fletcher penalty function:
    
min\\_x f(x) - dot(c(x),ys(x))

where

ys(x) := argmin\\_y 0.5 ||A(x)y - g(x)||^2\\_2 + σ c(x)^T y

and denote Ys the gradient of ys(x).

`FletcherPenaltyNLP(:: AbstractNLPModel, :: Number, :: Function)`
or
`FletcherPenaltyNLP(:: AbstractNLPModel; sigma_0 :: Real = 1.0, linear_system_solver :: Function = _solve_with_linear_operator)`

Notes:
- Evaluation of the obj, grad, objgrad functions evaluate functions from the orginial nlp.
These values are stored in *fx*, *cx*, *gx*.
- The value of the penalty vector *ys* is also stored.
- `linear_system_solver(nlp, x, rhs1, Union{rhs2,nothing})` is a function that successively solve
the two linear systems and returns the two solutions.

TODO:
- hprod
- write objgrad

Example:
fp_sos  = FletcherPenaltyNLP(nlp, 0.1, _solve_with_linear_operator)
"""
mutable struct FletcherPenaltyNLP{S <: AbstractFloat, T <: AbstractVector{S}} <: AbstractNLPModel

    meta     :: AbstractNLPModelMeta
    counters :: Counters
    nlp      :: AbstractNLPModel

    #Evaluation of the FletcherPenaltyNLP functions contains info on nlp:
    fx  :: Union{S}
    cx  :: Union{T}
    gx  :: Union{T}
    ys  :: Union{T}

    sigma :: Number
    linear_system_solver :: Function

    hessian_approx :: Int
    
end

function FletcherPenaltyNLP(nlp, sigma, linear_system_solver)
    x0=nlp.meta.x0
    S, T = eltype(nlp.meta.x0), typeof(nlp.meta.x0)
    hessian_approx = 2
    
    nvar = nlp.meta.nvar

    nnzh = nvar * (nvar + 1) / 2

    meta = NLPModelMeta(nvar, x0 = x0, nnzh = nnzh, 
                              minimize = true, islp = false, 
                              name = "Fletcher penalization of $(nlp.meta.name)")
    counters = Counters()
    return FletcherPenaltyNLP(meta, counters, nlp, 
                              NaN, S[], S[], S[], 
                              sigma, linear_system_solver, hessian_approx)
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

function FletcherPenaltyNLP(nlp                  :: AbstractNLPModel; 
                            sigma_0              :: Real = one(eltype(nlp.meta.x0)), 
                            linear_system_solver :: Function = _solve_with_linear_operator)
 return FletcherPenaltyNLP(nlp, sigma_0, linear_system_solver)
end

function obj(nlp ::  FletcherPenaltyNLP, x :: AbstractVector{T}) where {T <: AbstractFloat}

    f     = obj(nlp.nlp, x);  nlp.fx = f;
    c     = cons(nlp.nlp, x); nlp.cx = c;
    g     = grad(nlp.nlp, x); nlp.gx = g;
    sigma = nlp.sigma
    rhs1  = vcat(g, sigma * c)

    _sol1, _  = nlp.linear_system_solver(nlp, x, rhs1, nothing)

    gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    nlp.ys = ys
    fx     = f - dot(c, ys)

    return fx
end

function grad!(nlp ::  FletcherPenaltyNLP, x :: AbstractVector{T}, gx :: AbstractVector{T}) where {T <: AbstractFloat}

    c     = cons(nlp.nlp, x); nlp.cx = c;
    g     = grad(nlp.nlp, x); nlp.gx = g;
    sigma = nlp.sigma

    rhs1  = vcat(g, sigma * c)
    rhs2  = vcat(zeros(nlp.meta.nvar), c)

    _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

    gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    nlp.ys = ys
    v, w   = _sol2[1:nlp.meta.nvar], _sol2[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    Hsv    = hprod(nlp.nlp, x, ys, v, obj_weight = 1.0)
    Sstw   = hprod(nlp.nlp, x, w, gs; obj_weight = 0.0)
    Ysc    = Hsv - nlp.sigma * v - Sstw

    gx .= gs - Ysc

 return gx
end

function objgrad!(nlp :: FletcherPenaltyNLP, x :: AbstractVector{T}, gx :: AbstractVector{T}) where {T <: AbstractFloat}

    f     = obj(nlp.nlp, x);  nlp.fx = f;
    c     = cons(nlp.nlp, x); nlp.cx = c;
    g     = grad(nlp.nlp, x); nlp.gx = g;
    sigma = nlp.sigma

    rhs1  = vcat(g, sigma * c)
    rhs2  = vcat(zeros(nlp.meta.nvar), c)

    _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

    gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    nlp.ys = ys
    v, w   = _sol2[1:nlp.meta.nvar], _sol2[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    Hsv    = hprod(nlp.nlp, x, ys, v, obj_weight = 1.0)
    Sstw   = hprod(nlp.nlp, x, w, gs; obj_weight = 0.0)
    Ysc    = Hsv - nlp.sigma * v - Sstw

    fx  = f - dot(c, ys)
    gx .= gs - Ysc

 return fx, gx
end

"""
    hess_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function hess_structure!(nlp :: FletcherPenaltyNLP, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end


function hess_coord!(nlp :: FletcherPenaltyNLP, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)

  f     = obj(nlp.nlp, x);  nlp.fx = f;
  c     = cons(nlp.nlp, x); nlp.cx = c;
  g     = grad(nlp.nlp, x); nlp.gx = g;
  A     = jac(nlp.nlp, x)
  sigma = nlp.sigma

  rhs1  = vcat(g, sigma * c)
  rhs2  = vcat(zeros(nlp.meta.nvar), c)

  _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

  gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]

  Hs = hess(nlp.nlp, x, ys)
  In = Matrix(I, nlp.meta.nvar, nlp.meta.nvar)
  Im = Matrix(I, nlp.nlp.meta.ncon, nlp.nlp.meta.ncon)
  Pt = A' * inv(Matrix(A*A') + 1e-3 * Im) * A
  Hx = (In - Pt) * Hs - Hs * Pt + 2 * sigma * Pt

  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end

  return vals
end

function hess_coord!(nlp :: FletcherPenaltyNLP, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  #This is an unconstrained optimization problem
  return hess_coord!(nlp, x, vals; obj_weight = obj_weight)
end

function hprod!(nlp :: FletcherPenaltyNLP, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv  :: AbstractVector; obj_weight=1.0)
 return hprod!(nlp, x, v, Hv, obj_weight = obj_weight)
end

function hprod!(nlp :: FletcherPenaltyNLP, x :: AbstractVector, v :: AbstractVector, Hv  :: AbstractVector; obj_weight=1.0)
 @lencheck nlp.meta.nvar x v Hv
 increment!(nlp, :neval_hprod)

 f     = obj(nlp.nlp, x);  nlp.fx = f;
 c     = cons(nlp.nlp, x); nlp.cx = c;
 g     = grad(nlp.nlp, x); nlp.gx = g;
 sigma = nlp.sigma

 rhs1  = vcat(g, sigma * c)
 rhs2  = vcat(zeros(nlp.meta.nvar), c)

 _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

 gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]

 Hsv    = hprod(nlp.nlp, x, ys, v, obj_weight = 1.0)

 pt_rhs1 = vcat(v,   zeros(nlp.nlp.meta.ncon))
 pt_rhs2 = vcat(Hsv, zeros(nlp.nlp.meta.ncon))
 pt_sol1, pt_sol2  = nlp.linear_system_solver(nlp, x, pt_rhs1, pt_rhs2)
 Ptv   = v   - pt_sol1[1:nlp.meta.nvar]
 PtHsv = Hsv - pt_sol2[1:nlp.meta.nvar]
 HsPtv = hprod(nlp.nlp, x, ys, Ptv, obj_weight = 1.0)

 Hv .= Hsv - PtHsv - HsPtv + 2 * sigma * Ptv

 return Hv
end
