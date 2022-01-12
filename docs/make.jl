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
    "Hierarchical Models",
    "Custom Priors",
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

    return joinpath("tutorials", "$file.md")
end

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
    pages=["Home" => "index.md", "Tutorials" => md_files, "API reference" => "api.md"],
)

deploydocs(; repo="github.com/TuringLang/TuringGLM.jl", devbranch="main")
