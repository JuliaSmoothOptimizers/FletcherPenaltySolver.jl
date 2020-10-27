import NLPModels: obj, objgrad, objgrad!, grad!, grad, hess
"""
We consider here the implementation of Fletcher's exact penalty method for
the minimization problem:
min\\_x f(x) s.t. c(x) = 0

ys = argmin\\_y 0.5 ||A(x)y - g(x)||^2\\_2 + σ c(x)^T y
and we denote Ys its gradient.

`FletcherPenaltyNLP(:: AbstractNLPModelMeta, :: Counters, :: AbstractNLPModel, :: Number, :: Function)`
or
`FletcherPenaltyNLP(:: AbstractNLPModel; sigma_0 :: Number = 1.0, linear_system_solver :: Function = _solve_with_linear_operator)`

Notes:
- Evaluation of the obj, grad, objgrad functions evaluate functions from the orginial nlp.
These values are stored in *fx*, *cx*, *gx*.
- The value of the penalty vector *ys* is also stored.

TODO:
- hprod

Example:
fp_sos  = FletcherPenaltyNLP(NLPModelMeta(n), Counters(), nlp, 0.1, _solve_with_linear_operator)
"""
mutable struct FletcherPenaltyNLP <: AbstractNLPModel

    meta     :: AbstractNLPModelMeta
    counters :: Counters
    nlp      :: AbstractNLPModel

    #Evaluation of the FletcherPenaltyNLP functions contains info on nlp:
    fx  :: Union{Number, AbstractVector, Nothing}
    cx  :: Union{AbstractVector, Nothing}
    gx  :: Union{AbstractVector, Nothing}
    ys  :: Union{AbstractVector, Nothing}

    sigma :: Number
    linear_system_solver :: Function

    function FletcherPenaltyNLP(meta, counters, nlp, sigma, linear_system_solver)
        return new(meta, counters, nlp, nothing, nothing, nothing, nothing, sigma, linear_system_solver)
    end
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

function FletcherPenaltyNLP(nlp :: AbstractNLPModel; sigma_0 :: Number = 1.0, linear_system_solver :: Function = _solve_with_linear_operator)
 return FletcherPenaltyNLP(nlp.meta, nlp.counters, nlp, sigma_0, linear_system_solver)
end

function grad!(nlp ::  FletcherPenaltyNLP, x :: AbstractVector, gx :: AbstractVector)

    c     = cons(nlp.nlp, x); nlp.cx = c;
    g     = grad(nlp.nlp, x); nlp.gx = g;
    sigma = nlp.sigma

    rhs1  = vcat(g, sigma * c)
    rhs2  = vcat(zeros(nlp.meta.nvar), c)

    _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, rhs2)

    gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    nlp.ys = ys;
    v, w   = _sol2[1:nlp.meta.nvar], _sol2[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    Hsv    = hprod(nlp.nlp, x, ys, v, obj_weight = 1.0)
    Sstw   = hprod(nlp.nlp, x, w, gs; obj_weight = 0.0)
    Ysc    = Hsv - nlp.sigma * v - Sstw

    gx .= gs - Ysc

 return gx
end

function obj(nlp ::  FletcherPenaltyNLP, x :: AbstractVector)

    f     = obj(nlp.nlp, x);  nlp.fx = f;
    c     = cons(nlp.nlp, x); nlp.cx = c;
    g     = grad(nlp.nlp, x); nlp.gx = g;
    sigma = nlp.sigma
    rhs1  = vcat(g, sigma * c)

    _sol1, _sol2  = nlp.linear_system_solver(nlp, x, rhs1, nothing)

    gs, ys = _sol1[1:nlp.meta.nvar], _sol1[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]
    nlp.ys = ys
    fx     = f - dot(c, ys)

    return fx
end

function objgrad!(nlp ::  FletcherPenaltyNLP, x :: AbstractVector, gx :: AbstractVector)

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
