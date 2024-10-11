ENV["GKSwstype"] = "100"
using ADNLPModels
using Documenter
using Printf
using FletcherPenaltySolver

pages = [
  "Introduction" => "index.md",
  "Tutorial" => "tutorial.md",
  "Fine-tune FPS" => "fine-tuneFPS.md",
  "Reference" => "reference.md",
]

makedocs(
  modules = [FletcherPenaltySolver],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "FletcherPenaltySolver",
  pages = pages,
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/FletcherPenaltySolver.jl.git",
  push_preview = true,
  devbranch = "main",
)
