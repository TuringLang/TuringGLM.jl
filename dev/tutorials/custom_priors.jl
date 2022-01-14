### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 587f852c-f357-4260-ae87-a60eaf101d36
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

# ╔═╡ 206214e5-ab7a-4cae-bc85-7a77ed909c39
using CSV

# ╔═╡ 7f497bba-b0cc-40ef-a32f-4f69fa1cc355
using DataFrames

# ╔═╡ 178595bd-d701-47c6-a33c-4a33bad90a4f
using TuringGLM

# ╔═╡ 27c478b2-730f-11ec-1fbd-0b63384eaaf4
md"""
Let's cover the **Linear Regression** example with the `kidiq` dataset (Gelman & Hill, 2007), which is data from a survey of adult American women and their respective children.
Dated from 2007, it has 434 observations and 4 variables:

* `kid_score`: child's IQ
* `mom_hs`: binary/dummy (0 or 1) if the child's mother has a high school diploma
* `mom_iq`: mother's IQ
* `mom_age`: mother's age
"""

# ╔═╡ 041e7dca-b554-450c-9f43-79494e54e48e
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/kidiq.csv"

# ╔═╡ c0c65854-57af-488d-b80b-d37601e8024b
kidiq = CSV.read(download(url), DataFrame)

# ╔═╡ 31287120-a04c-46b2-ba06-6168c3435a78
md"""
Using `kid_score` as dependent variable and `mom_hs` along with `mom_iq` as independent variables with a moderation (interaction) effect:
"""

# ╔═╡ 1e4c7e9e-b32a-4eca-8621-d3b748d45d9e
fm = @formula(kid_score ~ mom_hs * mom_iq)

# ╔═╡ 0cd3dfce-3001-4c6e-b550-ca6964f19e35
md"""
Let's create our CustomPrior object.
No need for the third (auxiliary) prior for this model so we leave it as `nothing`:
"""

# ╔═╡ 9766b37c-55a0-4f74-924e-66b92eab7429
priors = CustomPrior(Normal(0, 2.5), Normal(10, 20), nothing);

# ╔═╡ 56498ac7-3476-42eb-9c12-078562fff51d
md"""
We instantiate our model with `turing_model` without specifying any model, thus the default model will be used: `Gaussian()`.
Notice that we are specifying the `priors` keyword argument:
"""

# ╔═╡ 99cbb309-393f-4a13-9454-1dee747c88a6
model = turing_model(fm, kidiq; priors);

# ╔═╡ 3e227e48-259c-40db-90e4-cd62afc307b2
chn = sample(model, NUTS(), 2_000);

# ╔═╡ 954f928a-5fcb-4c14-a74c-09bfe92f50bd
# hide
plot_chains(chn)

# ╔═╡ 9bb7a28f-a980-4cbc-9931-b38df79ae94b
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═587f852c-f357-4260-ae87-a60eaf101d36
# ╠═27c478b2-730f-11ec-1fbd-0b63384eaaf4
# ╠═206214e5-ab7a-4cae-bc85-7a77ed909c39
# ╠═7f497bba-b0cc-40ef-a32f-4f69fa1cc355
# ╠═041e7dca-b554-450c-9f43-79494e54e48e
# ╠═c0c65854-57af-488d-b80b-d37601e8024b
# ╠═178595bd-d701-47c6-a33c-4a33bad90a4f
# ╠═31287120-a04c-46b2-ba06-6168c3435a78
# ╠═1e4c7e9e-b32a-4eca-8621-d3b748d45d9e
# ╠═0cd3dfce-3001-4c6e-b550-ca6964f19e35
# ╠═9766b37c-55a0-4f74-924e-66b92eab7429
# ╠═56498ac7-3476-42eb-9c12-078562fff51d
# ╠═99cbb309-393f-4a13-9454-1dee747c88a6
# ╠═3e227e48-259c-40db-90e4-cd62afc307b2
# ╠═954f928a-5fcb-4c14-a74c-09bfe92f50bd
# ╠═9bb7a28f-a980-4cbc-9931-b38df79ae94b
