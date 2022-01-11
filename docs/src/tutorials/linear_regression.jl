### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ a4b92179-59f1-4406-a6d8-e2c844f63e95
# hideall
let
	pkg_dir = if "PKGDIR" in keys(ENV)
		ENV["PKGDIR"]
	else
		dirname(dirname(dirname(@__DIR__)))
	end
	docs_dir = joinpath(pkg_dir, "docs")

	using Pkg: Pkg
	Pkg.activate(docs_dir)
	Pkg.develop(path=pkg_dir)
	Pkg.instantiate()
end

# ╔═╡ 79cac787-9b01-4c0e-bb7d-f4555fcf6216
using CSV

# ╔═╡ 2d99b5d8-761c-489c-9bbe-bb48fd471ea5
using DataFrames

# ╔═╡ 4c99ba26-0041-4aaa-b224-44819dc72a87
using TuringGLM

# ╔═╡ 1346efe0-72f4-11ec-0251-5fac34b1eec8
md"""
Let's cover **Linear Regression** with a famous dataset called `kidiq` (Gelman & Hill, 2007), which is data from a survey of adult American women and their respective children.
Dated from 2007, it has 434 observations and 4 variables:

* `kid_score`: child's IQ
* `mom_hs`: binary/dummy (0 or 1) if the child's mother has a high school diploma
* `mom_iq`: mother's IQ
* `mom_age`: mother's age

For the purposes of this tutorial, we download the dataset from the TuringGLM repository:
"""

# ╔═╡ e0a8ebec-3b29-42c2-a406-47d99df64b68
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/kidiq.csv";

# ╔═╡ 65b092a5-e6c5-48a0-9058-39a5c0094afa
kidiq = CSV.read(download(url), DataFrame)

# ╔═╡ 2eb258a0-854b-4ea6-80f9-f88d868ec839
md"""
Using `kid_score` as dependent variable and `mom_hs` along with `mom_iq` as independent variables with a moderation (interaction) effect:
"""

# ╔═╡ a1031c0e-2237-409e-bf7d-65efb071f1da
fm = @formula(kid_score ~ mom_hs * mom_iq)

# ╔═╡ 43d63761-adf5-4a52-b996-4ad3adfb35d0
md"""
Next, we instantiate our model with `turing_model` without specifying any model, thus the default model will be used: `Gaussian()`:
"""

# ╔═╡ 55b91963-001e-4753-93a6-2fa64190f353
model = turing_model(fm, kidiq);

# ╔═╡ 92bec7a8-3524-4daa-9ef9-b58f2e59dea4
n_samples = 2_000;

# ╔═╡ 7935378b-9f7a-418c-9d51-e77e4a551bde
md"""
This model is a valid Turing model, which we can pass to the default `sample` function from Turing to get our parameter estimates.
We use the `NUTS` sampler with $n_samples samples.
"""

# ╔═╡ 3f26f821-0985-4e15-845d-4791c623a799
chn = sample(model, NUTS(), n_samples);

# ╔═╡ 4adf2c8d-3f05-4ceb-897c-ee2e1b156474
describe(chn)[1]

# ╔═╡ 28ef5f85-7645-4b26-ad71-b043ef141a50
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═a4b92179-59f1-4406-a6d8-e2c844f63e95
# ╠═1346efe0-72f4-11ec-0251-5fac34b1eec8
# ╠═79cac787-9b01-4c0e-bb7d-f4555fcf6216
# ╠═2d99b5d8-761c-489c-9bbe-bb48fd471ea5
# ╠═4c99ba26-0041-4aaa-b224-44819dc72a87
# ╠═e0a8ebec-3b29-42c2-a406-47d99df64b68
# ╠═65b092a5-e6c5-48a0-9058-39a5c0094afa
# ╠═2eb258a0-854b-4ea6-80f9-f88d868ec839
# ╠═a1031c0e-2237-409e-bf7d-65efb071f1da
# ╠═43d63761-adf5-4a52-b996-4ad3adfb35d0
# ╠═55b91963-001e-4753-93a6-2fa64190f353
# ╠═92bec7a8-3524-4daa-9ef9-b58f2e59dea4
# ╠═7935378b-9f7a-418c-9d51-e77e4a551bde
# ╠═3f26f821-0985-4e15-845d-4791c623a799
# ╠═4adf2c8d-3f05-4ceb-897c-ee2e1b156474
# ╠═28ef5f85-7645-4b26-ad71-b043ef141a50
