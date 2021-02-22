struct AlgoData{T <: Real}

    #Initialize, Update and Bound parameters of the penalized problem:
    σ_0      :: T
    σ_max    :: T
    σ_update :: T
    ρ_0      :: T
    ρ_max    :: T
    ρ_update :: T
    δ_0      :: T
    
    #Bound on the Lagrange multipliers
    yM       :: T
    
    #Algorithmic parameters
    Δ        :: T #expected decrease in feasibility between two iterations
    
    #Functions used in the algorithm
    linear_system_solver  :: Function
    
    unconstrained_solver  :: Function
    
    hessian_approx
    convex_subproblem :: Bool #Useful to set the `convex` option in Knitro
end

function AlgoData(T                    :: DataType;
                  σ_0                  :: Real     = one(T),
                  σ_max                :: Real     = 1/eps(T),
                  σ_update             :: Real     = T(1.15),
                  ρ_0                  :: Real     = one(T),
                  ρ_max                :: Real     = 1/eps(T),
                  ρ_update             :: Real     = T(1.15),
                  δ_0                  :: Real     = √eps(T),
                  yM                   :: Real     = typemax(T),
                  Δ                    :: Real     = T(0.95),
                  linear_system_solver :: Function = _solve_ldlt_factorization, #_solve_with_linear_operator,
                  unconstrained_solver :: Function = knitro,
                  hessian_approx       :: Int       = 2,
                  convex_subproblem    :: Bool      = false,
                  kwargs...)
                  
   return AlgoData(σ_0, σ_max, σ_update, ρ_0, ρ_max, ρ_update, δ_0, yM, Δ, 
                   linear_system_solver, unconstrained_solver, 
                   hessian_approx, convex_subproblem)
end

AlgoData(;kwargs...) = AlgoData(Float64;kwargs...)
