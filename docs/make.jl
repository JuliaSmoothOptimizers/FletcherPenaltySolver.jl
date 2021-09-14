using ADNLPModels
using Documenter
using Printf
using FletcherPenaltyNLPSolver

makedocs(
  modules = [FletcherPenaltyNLPSolver],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "FletcherPenaltyNLPSolver",
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(repo = "github.com/tmigot/FletcherPenaltyNLPSolver.git", devbranch = "main")
