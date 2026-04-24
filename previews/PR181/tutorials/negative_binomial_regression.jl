### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 19befc40-7302-11ec-2216-a533c6a003d4
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

# ╔═╡ 544e6e15-dce2-408b-aed2-b7aa7c16b669
using CSV

# ╔═╡ c2008c37-756f-43e1-beb1-9471ab054229
using DataFrames

# ╔═╡ a9b80331-2ca3-4c1e-beec-d3f7c9227e58
using TuringGLM

# ╔═╡ a8873af4-f4eb-4285-aabe-67abf9656570
md"""
For our example on **Negative Binomial Regression**, let's use a famous dataset called `roaches` (Gelman & Hill, 2007), which is data on the efficacy of a
pest management system at reducing the number of roaches in urban apartments.
It has 262 observations and the following variables:

* `y` -- number of roaches caught.
* `roach1` -- pretreatment number of roaches.
* `treatment` -- binary/dummy (0 or 1) for treatment indicator.
* `senior` -- binary/dummy (0 or 1) for only elderly residents in building.
* `exposure2` -- number of days for which the roach traps were used
"""

# ╔═╡ 5f29861c-cd0e-4f33-bf4e-3cb53faa1e7a
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/roaches.csv";

# ╔═╡ ac9084ba-9287-44df-843e-5ba33fde2055
roaches = CSV.read(download(url), DataFrame)

# ╔═╡ e715eda7-bcbf-4292-b95f-003816d47342
md"""
Using `y` as dependent variable and `roach1`, `treatment`, and `senior` as independent variables:
"""

# ╔═╡ fcec7550-4902-4907-a3a0-ef34b0f0cc21
fm = @formula(y ~ roach1 + treatment + senior)

# ╔═╡ 124aeb01-7661-4402-a2fa-77d2771b686c
md"""
We instantiate our model with `turing_model` passing a keyword argument `model=NegativeBinomial` to indicate that the model is a negative binomial regression:
"""

# ╔═╡ 7ea3a50d-3b0d-4cfd-b311-806d6ae59c1a
model = turing_model(fm, roaches; model=NegativeBinomial);

# ╔═╡ 597ce5d7-df3e-44e3-a154-e06e64894854
chn = sample(model, NUTS(), 2_000);

# ╔═╡ e8aec499-f48f-4e7f-93a6-ba38881ce147
# hide
plot_chains(chn)

# ╔═╡ abd3b2b0-af16-4895-9391-72cab5fda507
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═19befc40-7302-11ec-2216-a533c6a003d4
# ╠═a8873af4-f4eb-4285-aabe-67abf9656570
# ╠═544e6e15-dce2-408b-aed2-b7aa7c16b669
# ╠═c2008c37-756f-43e1-beb1-9471ab054229
# ╠═5f29861c-cd0e-4f33-bf4e-3cb53faa1e7a
# ╠═ac9084ba-9287-44df-843e-5ba33fde2055
# ╠═a9b80331-2ca3-4c1e-beec-d3f7c9227e58
# ╠═e715eda7-bcbf-4292-b95f-003816d47342
# ╠═fcec7550-4902-4907-a3a0-ef34b0f0cc21
# ╠═124aeb01-7661-4402-a2fa-77d2771b686c
# ╠═7ea3a50d-3b0d-4cfd-b311-806d6ae59c1a
# ╠═597ce5d7-df3e-44e3-a154-e06e64894854
# ╠═e8aec499-f48f-4e7f-93a6-ba38881ce147
# ╠═abd3b2b0-af16-4895-9391-72cab5fda507
