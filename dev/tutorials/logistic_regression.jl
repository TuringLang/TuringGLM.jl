### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 4e046622-7300-11ec-0a1d-974094a4752e
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

# ╔═╡ 544dfb44-b8ae-4c2c-a03b-3e14c88c1d66
using CSV

# ╔═╡ d6fa288d-acba-49a9-ba58-c7a871cca5ad
using DataFrames

# ╔═╡ 3061ae4d-49c4-4275-9505-f75aa8be23a7
using TuringGLM

# ╔═╡ 89586742-fbe8-4213-90d4-204f6b15d514
md"""
For our tutorial on **Logistic Regression**, let's use a famous dataset called `wells` (Gelman & Hill, 2007),
which is data from a survey of 3,200 residents in a small area of Bangladesh suffering from arsenic contamination of groundwater.
Respondents with elevated arsenic levels in their wells had been encouraged to switch their water source to a safe public or private well in the nearby area and the survey was conducted several years later to learn which of the affected residents had switched wells.
It has 3,200 observations and the following variables:

* `switch` -- binary/dummy (0 or 1) for well-switching.
* `arsenic` -- arsenic level in respondent's well.
* `dist` -- distance (meters) from the respondent's house to the nearest well with safe drinking water.
* `association` -- binary/dummy (0 or 1) if member(s) of household participate in community organizations.
* `educ` -- years of education (head of household).
"""

# ╔═╡ e0cf019d-2c2e-470b-a62a-25fbb9a22b4d
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/wells.csv";

# ╔═╡ 0de64701-70cf-4b1c-ae5a-e288e34a5e3c
wells = CSV.read(download(url), DataFrame)

# ╔═╡ ffb09bcb-3d02-455c-b153-345f4857fe9d
md"""
Using `switch` as dependent variable and `dist`, `arsenic`, `assoc`, and `educ` as independent variables:
"""

# ╔═╡ 07175d0e-7293-4095-95cc-91fe493a1aef
fm = @formula(switch ~ dist + arsenic + assoc + educ)

# ╔═╡ 2ebfa422-f8a5-44d3-8f2c-a34d7832d3f2
md"""
Now we instantiate our model with `turing_model` passing a third argument `Logistic()` to indicate that the model is a logistic regression:
"""

# ╔═╡ 1f1158ce-73f0-49fd-a48a-af3b36376030
model = turing_model(fm, wells, Logistic());

# ╔═╡ ebe074cb-7bad-4b52-9c9d-b9752af4bedd
chn = sample(model, NUTS(), 2_000);

# ╔═╡ d6499c1d-c2a1-4cc0-a8f4-c4ba86d5cf30
# hide
plot_chains(chn)

# ╔═╡ 2361e758-1b5b-4cb2-b7d5-8a03760befa6
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═4e046622-7300-11ec-0a1d-974094a4752e
# ╠═89586742-fbe8-4213-90d4-204f6b15d514
# ╠═544dfb44-b8ae-4c2c-a03b-3e14c88c1d66
# ╠═d6fa288d-acba-49a9-ba58-c7a871cca5ad
# ╠═3061ae4d-49c4-4275-9505-f75aa8be23a7
# ╠═e0cf019d-2c2e-470b-a62a-25fbb9a22b4d
# ╠═0de64701-70cf-4b1c-ae5a-e288e34a5e3c
# ╠═ffb09bcb-3d02-455c-b153-345f4857fe9d
# ╠═07175d0e-7293-4095-95cc-91fe493a1aef
# ╠═2ebfa422-f8a5-44d3-8f2c-a34d7832d3f2
# ╠═1f1158ce-73f0-49fd-a48a-af3b36376030
# ╠═ebe074cb-7bad-4b52-9c9d-b9752af4bedd
# ╠═d6499c1d-c2a1-4cc0-a8f4-c4ba86d5cf30
# ╠═2361e758-1b5b-4cb2-b7d5-8a03760befa6
