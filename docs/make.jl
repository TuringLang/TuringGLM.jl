using TuringGLM
using Documenter
using Pluto: Configuration.CompilerOptions
using PlutoStaticHTML

tutorials = [
    "Linear Regression",
    "Logistic Regression",
    "Poisson Regression",
    "Negative Binomial Regression",
    "Robust Regression",
    "Hierarchical Models",
    "Custom Priors",
]

include("build.jl")

build()
md_files = markdown_files()
T = [t => f for (t, f) in zip(tutorials, md_files)]

DocMeta.setdocmeta!(TuringGLM, :DocTestSetup, :(using TuringGLM); recursive=true)

makedocs(;
    modules=[TuringGLM],
    authors="Jose Storopoli <thestoropoli@gmail.com>, Rik Huijzer <t.h.huijzer@rug.nl>, and contributors",
    repo="https://github.com/TuringLang/TuringGLM.jl/blob/{commit}{path}#{line}",
    sitename="TuringGLM.jl",
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
        canonical="https://TuringLang.github.io/TuringGLM.jl",
        # Using MathJax3 since Pluto uses that engine too.
        mathengine=Documenter.MathJax3(),
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=["Home" => "index.md", "Tutorials" => T, "API reference" => "api.md"],
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")

# Useful for local development.
cd(pkgdir(TuringGLM))
