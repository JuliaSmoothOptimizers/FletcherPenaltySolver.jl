@testset "Testing FPS-solver" begin
    @testset "$foo with 1st approximation" for foo in [
      #unconstrained_nlp,
      #bound_constrained_nlp,
      #equality_constrained_nlp, # all but one, which is solved but lead to another solution.
    ]
      foo(nlp -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(1)))
    end

    @testset "$foo with 2nd approximation" for foo in [
        #unconstrained_nlp,
        #bound_constrained_nlp,
        #equality_constrained_nlp, # all but one, which is solved but lead to another solution.
      ]
        foo(nlp -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(2)))
      end
  
    @testset "Multiprecision tests with 1st approximation" begin
      for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
        #multiprecision_nlp((nlp; kwargs...) -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(1); kwargs...), ptype)
      end
    end
    @testset "Multiprecision tests with 2nd approximation" begin
      for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen] 
        #multiprecision_nlp((nlp; kwargs...) -> fps_solve(nlp, nlp.meta.x0, hessian_approx = Val(2); kwargs...), ptype)
      end
    end
  end

  #=
Precision Float16 for ptype unc: Error During Test at /home/tmigot/.julia/packages/SolverTest/imbD0/src/nlp/multiprecision.jl:19
  Got exception outside of a @test
  MethodError: no method matching StoppingMeta(::Float16, ::Float16, ::Float64, ::FletcherPenaltyNLPSolver.var"#Fptc#33"{Vector{Float16}, Vector{Float16}}, ::Stopping.var"#44#52"{FletcherPenaltyNLPSolver.var"#Fptc#33"{Vector{Float16}, Vector{Float16}}}, ::Vector{Float64}, ::Vector{Float64}, ::typeof(FletcherPenaltyNLPSolver.Fletcher_penalty_optimality_check), ::Bool, ::Float64, ::Float64, ::Int64, ::Dict{Symbol, Int64}, ::Int64, ::Int64, ::Float64, ::Int64, ::Float64, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Nothing, ::Stopping.var"#46#54")
  Closest candidates are:
    StoppingMeta(::TolType, ::TolType, ::TolType, ::Union{Function, CheckType}, ::Union{Function, CheckType}, ::CheckType, ::CheckType, ::Function, ::Bool, ::TolType, ::TolType, ::IntType, ::Dict{Symbol, Int64}, ::IntType, ::IntType, ::Float64, ::IntType, ::Float64, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::Bool, ::MUS, ::Function) where {TolType<:Number, CheckType, MUS, IntType<:Int64} at /home/tmigot/.julia/packages/Stopping/E9jNz/src/Stopping/StoppingMetamod.jl:71
    StoppingMeta(::CheckType, ::CheckType; atol, rtol, optimality0, optimality_check, recomp_tol, unbounded_threshold, unbounded_x, max_f, max_cntrs, max_eval, max_iter, max_time, start_time, meta_user_struct, user_check_func!, kwargs...) where CheckType at /home/tmigot/.julia/packages/Stopping/E9jNz/src/Stopping/StoppingMetamod.jl:123
  Stacktrace:
    [1] StoppingMeta(; atol::Float16, rtol::Float16, optimality0::Float64, tol_check::FletcherPenaltyNLPSolver.var"#Fptc#33"{Vector{Float16}, Vector{Float16}}, tol_check_neg::Stopping.var"#44#52"{FletcherPenaltyNLPSolver.var"#Fptc#33"{Vector{Float16}, Vector{Float16}}}, optimality_check::Function, recomp_tol::Bool, unbounded_threshold::Float64, unbounded_x::Float64, max_f::Int64, max_cntrs::Dict{Symbol, Int64}, max_eval::Int64, max_iter::Int64, max_time::Float64, start_time::Float64, meta_user_struct::Nothing, user_check_func!::Stopping.var"#46#54", kwargs::Base.Iterators.Pairs{Symbol, Val{1}, Tuple{Symbol}, NamedTuple{(:hessian_approx,), Tuple{Val{1}}}})
      @ Stopping ~/.julia/packages/Stopping/E9jNz/src/Stopping/StoppingMetamod.jl:225
    [2] NLPStopping(pb::ADNLPModel, current_state::NLPAtX{Vector{Float16}, Vector{Float16}, Matrix{Float16}}; main_stp::VoidStopping{Any, StoppingMeta, StopRemoteControl, GenericState, Nothing, VoidListofStates}, list::VoidListofStates, user_struct::Dict{Any, Any}, kwargs::Base.Iterators.Pairs{Symbol, Any, NTuple{6, Symbol}, NamedTuple{(:optimality_check, :atol, :rtol, :tol_check, :max_cntrs, :hessian_approx), Tuple{typeof(FletcherPenaltyNLPSolver.Fletcher_penalty_optimality_check), Float16, Float16, FletcherPenaltyNLPSolver.var"#Fptc#33"{Vector{Float16}, Vector{Float16}}, Dict{Symbol, Int64}, Val{1}}}})
      @ Stopping ~/.julia/packages/Stopping/E9jNz/src/Stopping/NLPStoppingmod.jl:110
    [3] fps_solve(nlp::ADNLPModel, x0::Vector{Float16}; kwargs::Base.Iterators.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:hessian_approx, :atol, :rtol), Tuple{Val{1}, Float16, Float16}}})
      @ FletcherPenaltyNLPSolver ~/cvs/FletcherPenaltyNLPSolver/src/FletcherPenaltyNLPSolver.jl:117
    [4] (::var"#3#7"{var"#3#4#8"})(nlp::ADNLPModel; kwargs::Base.Iterators.Pairs{Symbol, Float16, Tuple{Symbol, Symbol}, NamedTuple{(:atol, :rtol), Tuple{Float16, Float16}}})
      @ Main ~/cvs/FletcherPenaltyNLPSolver/test/solvertest.jl:20
    [5] (::SolverTest.var"#64#68"{var"#3#7"{var"#3#4#8"}, Float16, ADNLPModel})()
      @ SolverTest ~/.julia/packages/SolverTest/imbD0/src/nlp/multiprecision.jl:42
    [6] with_logstate(f::Function, logstate::Any)
      @ Base.CoreLogging ./logging.jl:491
    [7] with_logger
      @ ./logging.jl:603 [inlined]
    [8] macro expansion
      @ ~/.julia/packages/SolverTest/imbD0/src/nlp/multiprecision.jl:41 [inlined]
    [9] macro expansion
      @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/Test/src/Test.jl:1226 [inlined]
   [10] multiprecision_nlp(solver::Function, ptype::Symbol)
      @ SolverTest ~/.julia/packages/SolverTest/imbD0/src/nlp/multiprecision.jl:19
   [11] macro expansion
      @ ~/cvs/FletcherPenaltyNLPSolver/test/solvertest.jl:20 [inlined]
   [12] macro expansion
      @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/Test/src/Test.jl:1151 [inlined]
   [13] macro expansion
      @ ~/cvs/FletcherPenaltyNLPSolver/test/solvertest.jl:19 [inlined]
   [14] macro expansion
      @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/Test/src/Test.jl:1151 [inlined]
   [15] top-level scope
      @ ~/cvs/FletcherPenaltyNLPSolver/test/solvertest.jl:2
   [16] include(fname::String)
      @ Base.MainInclude ./client.jl:444
   [17] top-level scope
      @ REPL[8]:1
   [18] eval
      @ ./boot.jl:360 [inlined]
   [19] eval_user_input(ast::Any, backend::REPL.REPLBackend)
      @ REPL /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/REPL/src/REPL.jl:139
   [20] repl_backend_loop(backend::REPL.REPLBackend)
      @ REPL /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/REPL/src/REPL.jl:200
   [21] start_repl_backend(backend::REPL.REPLBackend, consumer::Any)
      @ REPL /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/REPL/src/REPL.jl:185
   [22] run_repl(repl::REPL.AbstractREPL, consumer::Any; backend_on_current_task::Bool)
      @ REPL /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/REPL/src/REPL.jl:317
   [23] run_repl(repl::REPL.AbstractREPL, consumer::Any)
      @ REPL /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/REPL/src/REPL.jl:305
   [24] (::Base.var"#874#876"{Bool, Bool, Bool})(REPL::Module)
      @ Base ./client.jl:387
   [25] #invokelatest#2
      @ ./essentials.jl:708 [inlined]
   [26] invokelatest
      @ ./essentials.jl:706 [inlined]
   [27] run_main_repl(interactive::Bool, quiet::Bool, banner::Bool, history_file::Bool, color_set::Bool)
      @ Base ./client.jl:372
   [28] exec_options(opts::Base.JLOptions)
      @ Base ./client.jl:302
   [29] _start()
      @ Base ./client.jl:485
  =#