---
title: 'FletcherPenaltyNLPSolver.jl: A Julia implementation of Fletcher's penalty method for nonlinear optimization models'
tags:
  - Julia
  - nonlinear optimization
  - numerical optimization
  - large-scale optimization
  - constrained optimization
  - nonlinear programming
authors:
  - name: Tangi Migot^[corresponding author]
    orcid: 0000-0001-7729-2513
    affiliation: 1
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
  - name: Abel Soares Siqueira
    orcid: 0000-0003-4451-281X
    affiliation: 2
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
   index: 1
 - name: Netherlands eScience Center, Amsterdam, NL
   index: 2
date: 14 April 2022
bibliography: paper.bib

---

# Summary

`FletcherPenaltyNLPSolver.jl` is a new Julia [@bezanson2017julia] implementation of Fletcher's penalty method, introduced by @fletcher1970class, for nonlinear optimization models
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) \quad \text{subject to} \quad h(x) = 0, l \leq x \leq u,
\end{equation}
where  $f:\mathbb{R}^n \rightarrow \mathbb{R}$ and  $h:\mathbb{R}^n \rightarrow \mathbb{R}^m$ are twice continuously differentiable.
Note that this formulation includes inequality constraints as they can be reformulated as bounds using slack variables.
Fletcher's penalty method is an iterative method that aims to compute a local minimum of \eqref{eq:nlp} using first and second-order derivatives. 
Our implementation follows the analysis in [@estrin2020implementingequality] and [@estrin2020implementinggeneral] showing how to efficiently implement the method.
Our initial motivation for developing `FletcherPenaltyNLPSolver.jl` is to solve PDE-constrained optimization problems that are very large-scale problems.

Fletcher's penalty method replaces \eqref{eq:nlp} with a parametric bound-constrained optimization problem via Fletcher's penalty function, whose local minimum can be found by any solver designed for such problems.
This function is also smooth under classical assumptions, and the penalty function is exact, i.e. local minimizers of \eqref{eq:nlp} are minimizers of the penalty function for all values of the parameters sufficiently large. The main computational kernel for evaluating the penalty function and its derivatives is the solution of a certain saddle-point system.
If the system matrix is available explicitly, we can factorize it once and reuse the factors to evaluate the penalty function and its derivatives. The penalty function can also be adapted to be factorization-free by solving the linear system iteratively.

`FletcherPenaltyNLPSolver.jl` is built upon the JuliaSmoothOptimizers (JSO) tools [@jso]. JSO is an academic organization containing a collection of Julia packages for nonlinear optimization software development, testing, and benchmarking. It provides tools for building models, accessing problems repositories, and solving subproblems. `FletcherPenaltyNLPSolver.jl` takes as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl` [@orban-siqueira-nlpmodels-2020], a flexible data type to evaluate objective and constraints, their derivatives, and to provide any information that a solver might request from a model. The user can hand-code derivatives, use automatic differentiation, or use JSO-interfaces to classical mathematical optimization modeling languages such as AMPL [@fourer2003ampl], CUTEst [@cutest], or JuMP [@jump]. 

Internally, `FletcherPenaltyNLPSolver.jl` combines cutting-edge numerical linear algebra solvers. The evaluation of the derivatives of the penalized subproblem relies on solving least squares and least norm problems.
Our factorization-free implementation uses iterative methods for these two families of problems. We use the package `Krylov.jl` [@montoison-orban-krylov-2020], which provides more than 25 implementations of standard and novel Krylov methods, and they all can be used with Nvidia GPU via CUDA.jl [@besard2018juliagpu]. Note that the package also provides the possibility to use direct methods using the sparse factorization of a symmetric and quasi-definite matrix via `LDLFactorizations.jl` [@orban-ldlfactorizations-2020], or the well-known Fortran code `MA57` [@duff-2004] from the @HSL, via `HSL.jl` [@orban-hsl-2021].
The optimization of the subproblem can be carried out by any solver designed for bound-constraint problems. By default, we use the implementation of TRON from `JSOSolvers.jl` [@orban-siqueira-jsosolvers-2021] which is also factorization-free, but any JSO-compliant solver could be of use.

One of the significant advantages of our implementation is that it is factorization-free, i.e., it uses second-order information via Hessian-vector products but does not need access to the Hessian as an explicit matrix.
This makes `FletcherPenaltyNLPSolver.jl` a valuable asset for large-scale problems, for instance, to solve PDE-constrained optimization problems [@migot-orban-siqueira-pdenlpmodels-2021].

# Statement of need

Julia's JIT compiler is attractive for the design of efficient scientific computing software, and, in particular, mathematical optimization [@lubin2015computing], and has become a natural choice for developing new solvers.

There already exist ways to solve \eqref{eq:nlp} in Julia.
If \eqref{eq:nlp} is amenable to being modeled in `JuMP` [@jump], the model may be passed to state-of-the-art solvers, implemented in low-level compiled languages, via wrappers thanks to Julia's native interoperability with such languages.
However, interfaces to low-level languages have limitations that pure Julia implementations do not have, including the ability to apply solvers with various arithmetic types.
`Optim.jl` [@mogensen2018optim] implements a factorization-based pure Julia primal-dual interior-point method for problems with both equality and inequality constraints modeled after Artlelys Knitro [@byrd2006k] and Ipopt [@wachter2006implementation].
`Percival.jl` [@percival-jl] is a factorization-free pure Julia implementation of an augmented Lagrangian method for problems with both equality and inequality constraints based on bound-constrained subproblems.
`DCISolver.jl` [@migot2022dcisolver] is also an implementation for large-scale optimization, but in the current state does not handle inequality constraints and is not factorization-free.

`FletcherPenaltyNLPSolver.jl` can solve large-scale problems and can be benchmarked easily against other JSO-compliant solvers using `SolverBenchmark.jl` [@orban-siqueira-solverbenchmark-2020].
We include below performance profiles [@dolan2002benchmarking] of `FletcherPenaltyNLPSolver.jl` against Ipopt on 82 problems from CUTEst [@cutest] with up to 10,000 variables and 10,000 constraints. 

<!--
----------------
Ipopt solved 72 problems (88%) successfully, which is one more than DCI. Without explaining performance profiles in full detail, the plot on the left shows that Ipopt is the fastest on 20 of the problems (28%), while DCI is the fastest on 51 of the problems (72%) among the 71 problems solved by both solvers. The plot on the right shows that Ipopt used fewer evaluations of objective and constraint functions on 50 of the problems (70%), DCI used fewer evaluations on 17 of the problems (24%), and there was a tie in the number of evaluations on 4 problems (6%).
----------------
Overall, this performance profile is very encouraging for such a young implementation.
The package's documentation includes more extensive benchmarks on classical test sets showing that `FletcherPenaltyNLPSolver.jl` is also competitive with Artelys Knitro.

illustrating that `FletcherPenaltyNLPSolver.jl` is a fast and stable alternative to a state of the art solver

NOTE: Putting the code is too long
```
include("make_problems_list.jl") # setup a file `list_problems.dat` with problem names
include("benchmark.jl") # run the benchmark and store the result in `ipopt_dcildl_82.jld2`
include("figures.jl") # make the figure
```

![](ipopt_dcildl_fps_82.png){ width=100% }
-->

# Acknowledgements

Tangi Migot is supported by IVADO and the Canada First Research Excellence Fund / Apogée,
and Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
