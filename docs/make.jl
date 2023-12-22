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
    authors="Jose Storopoli <jose@storopoli.io>, Rik Huijzer <t.h.huijzer@rug.nl>, and contributors",
    sitename="TuringGLM.jl",
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
        canonical="https://TuringLang.github.io/TuringGLM.jl",
        # Using MathJax3 since Pluto uses that engine too.
        mathengine=Documenter.MathJax3(),
        prettyurls=get(ENV, "CI", "false") == "true",
        size_threshold=600 * 2^10, # 600 KiB
        size_threshold_warn=200 * 2^10, # 200 KiB
    ),
    pages=["Home" => "index.md", "Tutorials" => T, "API reference" => "api.md"],
    linkcheck=true,
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")

# Useful for local development.
cd(pkgdir(TuringGLM))
