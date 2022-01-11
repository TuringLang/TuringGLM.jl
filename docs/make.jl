using TuringGLM
using Documenter
using PlutoStaticHTML

DocMeta.setdocmeta!(TuringGLM, :DocTestSetup, :(using TuringGLM); recursive=true)

# PlutoStaticHTML opens the notebook in a tempdir, so this is a hack to pass the correct dir in again.
ENV["PKGDIR"] = string(pkgdir(TuringGLM))

# Run all Pluto notebooks (".jl" files) in `tutorials_dir` and write outputs to HTML files.
tutorials_dir = joinpath(pkgdir(TuringGLM), "docs", "src", "tutorials")
parallel_build(BuildOptions(tutorials_dir))

tutorials = [
    "Linear Regression",
    "Logistic Regression",
    "Poisson Regression",
    "Negative Binomial Regression",
    "Robust Regression",
    # "Hierarchical Models",
    # "Custom Priors"
]

# Generate tutorials Markdown files which can be read by Documenter.jl
md_files = map(tutorials) do tutorial
    file = lowercase(replace(tutorial, " " => '_'))

    from = joinpath(tutorials_dir, "$file.html")
    html = read(from, String)

    md = """
        # $tutorial

        ```@eval
        # Auto generated file. Do not modify.
        ```

        ```@raw html
        $html
        ```
        """

    to = joinpath(tutorials_dir, "$file.md")
    println("Writing $to")
    write(to, md)
end

makedocs(;
    modules=[TuringGLM],
    authors="Jose Storopoli <thestoropoli@gmail.com> and contributors",
    repo="https://github.com/TuringLang/TuringGLM.jl/blob/{commit}{path}#{line}",
    sitename="TuringGLM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TuringLang.github.io/TuringGLM.jl",
        assets=["assets/favicon.ico"],
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
            "tutorials/custom_priors.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")
