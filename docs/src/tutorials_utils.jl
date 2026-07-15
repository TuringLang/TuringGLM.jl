using CairoMakie
using FlexiChains
using TuringGLM

function plot_chains(chns)
    param_keys = filter(k -> k isa FlexiChains.Parameter, collect(keys(chns)))

    names_and_values = Pair{String,Matrix{Float64}}[]
    for k in param_keys
        name = string(FlexiChains.get_name(k))
        data = getindex(chns, k)
        if eltype(parent(data)) <: AbstractVector
            stacked = getindex(chns, k; stack=true)
            for j in axes(stacked, 3)
                push!(names_and_values, "$name[$j]" => Array(stacked[:, :, j]))
            end
        else
            push!(names_and_values, name => Array(data))
        end
    end

    n_chains = FlexiChains.nchains(chns)
    n_samples = FlexiChains.niters(chns)

    fig = Figure(; resolution=(1_000, 800))

    for (i, (name, values)) in enumerate(names_and_values)
        ax = Axis(fig[i, 1]; ylabel=name)
        for chain in 1:n_chains
            lines!(ax, 1:n_samples, values[:, chain]; label=string(chain))
        end

        hideydecorations!(ax; label=false)
        if i < length(names_and_values)
            hidexdecorations!(ax; grid=false)
        else
            ax.xlabel = "Iteration"
        end
    end

    for (i, (name, values)) in enumerate(names_and_values)
        ax = Axis(fig[i, 2]; ylabel=name)
        for chain in 1:n_chains
            density!(ax, values[:, chain]; label=string(chain))
        end

        hideydecorations!(ax)
        if i == length(names_and_values)
            ax.xlabel = "Parameter estimate"
        end
    end

    return fig
end
