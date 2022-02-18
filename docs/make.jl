ENV["GKSwstype"] = "100"
using ADNLPModels
using Documenter
using Printf
using FletcherPenaltyNLPSolver
using Literate

EXAMPLE = joinpath(@__DIR__, "assets", "example.jl")
OUTPUT = joinpath(@__DIR__, "src")

# Generate markdown
binder_badge = "# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tmigot/FletcherPenaltyNLPSolver/gh-pages?labpath=dev%2Fexample.ipynb)"
function preprocess_docs(content)
  return string(binder_badge, "\n\n", content)
end

Literate.markdown(EXAMPLE, OUTPUT; preprocess = preprocess_docs, codefence = "```julia" => "```")

link_to_env = "# The environment used in this tutorial is the following [Project.toml](https://github.com/tmigot/FletcherPenaltyNLPSolver/blob/gh-pages/Project.toml) and [Manifest.toml](https://github.com/tmigot/FletcherPenaltyNLPSolver/blob/gh-pages/Manifest.toml). "
function preprocess_notebook(content)
  return string(link_to_env, "\n\n", content)
end
Literate.notebook(EXAMPLE, OUTPUT; preprocess = preprocess_notebook)
Literate.script(EXAMPLE, OUTPUT)

pages = [
  "Introduction" => "index.md",
  "Tutorial" => "tutorial.md",
  # "Benchmark" => "benchmark.md",
  "Fine-tune FPS" => "fine-tuneFPS.md",
  "Large-scale example" => "example.md",
  "Reference" => "reference.md",
]

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
  pages = pages,
)

deploydocs(
  repo = "github.com/tmigot/FletcherPenaltyNLPSolver.git",
  push_preview = true,
  devbranch = "main",
)
