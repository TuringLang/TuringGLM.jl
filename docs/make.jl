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
md_files = generate_markdown_files()

DocMeta.setdocmeta!(TuringGLM, :DocTestSetup, :(using TuringGLM); recursive=true)

makedocs(;
    modules=[TuringGLM],
    authors="Jose Storopoli <thestoropoli@gmail.com>, Rik Huijzer <t.h.huijzer@rug.nl>, and contributors",
    repo="https://github.com/TuringLang/TuringGLM.jl/blob/{commit}{path}#{line}",
    sitename="TuringGLM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TuringLang.github.io/TuringGLM.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=["Home" => "index.md", "Tutorials" => md_files, "API reference" => "api.md"],
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")

# For local development.
cd(pkgdir(TuringGLM))
