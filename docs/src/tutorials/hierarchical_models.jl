### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 9188cab0-7304-11ec-24ed-8d1d5505988b
# hideall
let
    docs_dir = dirname(dirname(@__DIR__))
    pkg_dir = dirname(docs_dir)

    using Pkg: Pkg
    Pkg.activate(docs_dir)
    Pkg.develop(; path=pkg_dir)
    Pkg.instantiate()

    # Putting the include here to avoid Pluto getting confused about cell order.
    include(joinpath(docs_dir, "src", "tutorials_utils.jl"))
end;

# ╔═╡ 38fef837-ff32-4f38-83b4-cf1ab07d2b1e
using CSV

# ╔═╡ 79253973-b0cd-4b57-92dc-b708577f3a26
using DataFrames

# ╔═╡ e06ca210-b517-41f2-986c-c20c95ab7fb4
using TuringGLM

# ╔═╡ 635eae65-3d2c-4f4b-8987-6d93534bbcfd
md"""
Currently, TuringGLM only supports hierarchical models with a single random-intercept.
This is done by using the `(1 | group)` inside the `@formula` macro.

For our **Hierarchical Model** example, let's use a famous dataset called `cheese` (Boatwright, McCulloch & Rossi, 1999), which is data from cheese ratings.
A group of 10 rural and 10 urban raters rated 4 types of different cheeses (A, B, C and D) in two samples.
So we have $4 \cdot 20 \cdot 2 = 160$ observations and 4 variables:

* `cheese`: type of cheese from `A` to `D`
* `rater`: id of the rater from `1` to `10`
* `background`: type of rater, either `rural` or `urban`
* `y`: rating of the cheese
"""

# ╔═╡ 658a690a-bbec-4b80-9fb1-accf65e15832
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/cheese.csv";

# ╔═╡ 64459b91-4e2b-4e52-92ac-bda2f2377a0f
cheese = CSV.read(download(url), DataFrame)

# ╔═╡ 0f38a9fb-20e8-4ff0-8617-add48850892e
md"""
Using `y` as dependent variable and `background` is independent variable with a varying-intercept per `cheese` type:
"""

# ╔═╡ e2cc4deb-cb7a-45df-86b5-6bb47a7ed35e
fm = @formula(y ~ (1 | cheese) + background)

# ╔═╡ 934b7ec9-d4b3-435c-876f-49215dca8809
md"""
We instantiate our model with `turing_model` without specifying any model, thus the default model will be used (`model=Gaussian`):
"""

# ╔═╡ 5fc736a2-068e-4deb-9a77-bd21d93f6f32
model = turing_model(fm, cheese);

# ╔═╡ f34064fe-fe17-45e6-af79-0a727d408ac0
chn = sample(model, NUTS(), 2_000);

# ╔═╡ ae013cfb-1634-4341-88d0-ca68bc29dab7
# hide
plot_chains(chn)

# ╔═╡ 30622375-7018-45f6-ba84-0171c40b9c3e
md"""
## References

Boatwright, P., McCulloch, R., & Rossi, P. (1999).
Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model.
Journal of the American Statistical Association, 94(448), 1063–1073.
"""

# ╔═╡ Cell order:
# ╠═9188cab0-7304-11ec-24ed-8d1d5505988b
# ╠═635eae65-3d2c-4f4b-8987-6d93534bbcfd
# ╠═38fef837-ff32-4f38-83b4-cf1ab07d2b1e
# ╠═79253973-b0cd-4b57-92dc-b708577f3a26
# ╠═658a690a-bbec-4b80-9fb1-accf65e15832
# ╠═64459b91-4e2b-4e52-92ac-bda2f2377a0f
# ╠═e06ca210-b517-41f2-986c-c20c95ab7fb4
# ╠═0f38a9fb-20e8-4ff0-8617-add48850892e
# ╠═e2cc4deb-cb7a-45df-86b5-6bb47a7ed35e
# ╠═934b7ec9-d4b3-435c-876f-49215dca8809
# ╠═5fc736a2-068e-4deb-9a77-bd21d93f6f32
# ╠═f34064fe-fe17-45e6-af79-0a727d408ac0
# ╠═ae013cfb-1634-4341-88d0-ca68bc29dab7
# ╠═30622375-7018-45f6-ba84-0171c40b9c3e
