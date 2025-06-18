# TuringGLM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.github.io/TuringGLM.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.github.io/TuringGLM.jl/dev)
[![Build Status](https://github.com/TuringLang/TuringGLM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/TuringGLM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TuringLang/TuringGLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/TuringGLM.jl)
[![Coverage](https://coveralls.io/repos/github/TuringLang/TuringGLM.jl/badge.svg?branch=main)](https://coveralls.io/github/TuringLang/TuringGLM.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

TuringGLM makes easy to specify Bayesian **G**eneralized **L**inear **M**odels using the formula syntax and returns an instantiated [Turing](https://github.com/TuringLang/Turing.jl) model.

Heavily inspired by [brms](https://github.com/paul-buerkner/brms/) (uses RStan or CmdStanR) and [bambi](https://github.com/bambinos/bambi) (uses PyMC3).

The `@formula` macro is extended from [`StatsModels.jl`](https://github.com/JuliaStats/StatsModels.jl) along with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl) for the random-effects (a.k.a. group-level predictors).

<p align="center">
  <a href="https://www.youtube.com/watch?v=Cn8qVHo21EQ">
        <img src="https://img.youtube.com/vi/Cn8qVHo21EQ/0.jpg" alt="YouTube">
  </a>
</p>
