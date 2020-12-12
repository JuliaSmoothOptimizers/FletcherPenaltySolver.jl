function _solve_system_dense(nlp  :: FletcherPenaltyNLP,
                             x    :: AbstractVector{T},
                             rhs1 :: AbstractVector{T},
                             rhs2 :: Union{AbstractVector{T},Nothing};
                             kwargs...)  where T <: AbstractFloat

  A =  NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  In = diagm(0 => ones(nlp.meta.nvar))
  Im = diagm(0 => ones(nlp.nlp.meta.ncon))
  M = [In A'; A -nlp.δ*Im] #expensive

  sol1 = M \ rhs1

  if rhs2 != nothing
      sol2 = M \ rhs2
  else
      sol2 = nothing
  end

  return sol1, sol2
end

function _solve_with_linear_operator(nlp  :: FletcherPenaltyNLP,
                                     x    :: AbstractVector{T},
                                     rhs1 :: AbstractVector{T},
                                     rhs2 :: Union{AbstractVector{T},Nothing};
                                     _linear_system_solver :: Function = cg,
                                     kwargs...)  where T <: AbstractFloat

    #size(A) : nlp.nlp.meta.ncon x nlp.nlp.meta.nvar
    n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
    nn = nlp.nlp.meta.ncon + nlp.nlp.meta.nvar
    #Tanj: Would it be beneficial to have a jjtprod returning Jv and Jtv ?
    Mp(v) = vcat( v[1 : n] + jtprod(nlp.nlp, x, v[n + 1 : nn]),
                 jprod(nlp.nlp, x, v[1 : n]) - nlp.δ * v[n + 1 : nn])
    #LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
    opM = LinearOperator(T, nn, nn, true, true, v->Mp(v), w->Mp(w), u->Mp(u))

    (sol1, stats1)  = _linear_system_solver(opM, rhs1; kwargs...)
    if !stats1.solved
        @warn "Failed solving linear system with $(_linear_system_solver)."
    end

    if rhs2 != nothing
        (sol2, stats2)  = _linear_system_solver(opM, rhs2; kwargs...)
        if !stats2.solved
            @warn "Failed solving linear system with $(_linear_system_solver)."
        end
    else
        sol2 = nothing
    end

    return sol1, sol2
end

function _solve_system_factorization_eigenvalue(nlp  :: FletcherPenaltyNLP,
                                                x    :: AbstractVector{T},
                                                rhs1 :: AbstractVector{T},
                                                rhs2 :: Union{AbstractVector{T},Nothing};
                                                kwargs...)  where T <: AbstractFloat

        A =  NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
        In = diagm(0 => ones(nlp.meta.nvar))
        Im = diagm(0 => ones(nlp.nlp.meta.ncon))
        M = [In A'; A -nlp.δ*Im] #expensive
        
        O, Δ = eigen(M)#eigvecs(M), eigvals(M)
        # Boost negative values of Δ to 1e-8
        D = Δ .+ max.((1e-8 .- Δ), 0.0)

        sol1 = O*diagm(1.0 ./ D)*O'*rhs1

        if rhs2 != nothing
            sol2 = O*diagm(1.0 ./ D)*O'*rhs2
        else
            sol2 = nothing
        end

  return sol1, sol2
end

function _solve_system_factorization_lu(nlp  :: FletcherPenaltyNLP,
                                        x    :: AbstractVector{T},
                                        rhs1 :: AbstractVector{T},
                                        rhs2 :: Union{AbstractVector{T},Nothing};
                                        kwargs...) where T <: AbstractFloat

        n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
        A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
        In = Matrix{T}(I, n, n) #spdiagm(0 => ones(nlp.meta.nvar)) ?
        Im = Matrix{T}(I, ncon, ncon)
        M = [In A'; A -nlp.δ*Im] #expensive

        LU = lu(M)

        sol1 = LU \ rhs1

        if rhs2 != nothing
            sol2 = LU \ rhs2
        else
            sol2 = nothing
        end

  return sol1, sol2
end
