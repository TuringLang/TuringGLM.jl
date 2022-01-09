using TuringGLM
using Documenter

DocMeta.setdocmeta!(TuringGLM, :DocTestSetup, :(using TuringGLM); recursive=true)

makedocs(;
    modules=[TuringGLM],
    authors="Jose Storopoli <thestoropoli@gmail.com> and contributors",
    repo="https://github.com/TuringLang/TuringGLM.jl/blob/{commit}{path}#{line}",
    sitename="TuringGLM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TuringLang.github.io/TuringGLM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/linear_regression.md",
            "tutorials/logistic_regression.md",
            "tutorials/poisson_regression.md",
            "tutorials/negativebinomial_regression.md",
            "tutorials/robust_regression.md",
            "tutorials/hierarchical_models.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")
