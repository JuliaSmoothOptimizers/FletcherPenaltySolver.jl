using ADNLPModels
using Documenter
using Printf
using FletcherPenaltyNLPSolver

pages = ["Introduction" => "index.md", "Reference" => "reference.md"]

makedocs(
  sitename = "FletcherPenaltyNLPSolver",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [FletcherPenaltyNLPSolver],
  pages = pages,
)

deploydocs(repo = "github.com/tmigot/FletcherPenaltyNLPSolver.git", push_preview = true)
