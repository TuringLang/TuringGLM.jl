tutorials_dir = joinpath(pkgdir(TuringGLM), "docs", "src", "tutorials")

"""Run all Pluto notebooks (".jl" files) in `tutorials_dir` and write outputs to HTML files."""
function build()
    println("Building tutorials")
    # Evaluate notebooks in the same process to avoid having to recompile from scratch each time.
    use_distributed = false
    bopts = BuildOptions(tutorials_dir; use_distributed)
    parallel_build(bopts)
    return nothing
end

"Generate tutorials Markdown files which can be read by Documenter.jl"
function generate_markdown_files()
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
    return md_files
end
