### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ cea125d8-7303-11ec-3f43-6b79e00bca6a
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

# ╔═╡ e0b79bde-f1c1-4256-866e-eb0649e77cb7
using CSV

# ╔═╡ 5a7e2bbe-ded0-4ce4-b24c-6b4387a7e80d
using DataFrames

# ╔═╡ 6002455d-45ba-42f4-9601-03754c2906d1
using TuringGLM

# ╔═╡ cd2dc745-d2bf-4960-854d-bad17cf767ab
md"""
For the **Robust Regression** with Student-$t$ distribution as the likelihood, we'll use a famous dataset called `kidiq` (Gelman & Hill, 2007), which is data from a survey of adult American women and their respective children.
Dated from 2007, it has 434 observations and 4 variables:

* `kid_score`: child's IQ
* `mom_hs`: binary/dummy (0 or 1) if the child's mother has a high school diploma
* `mom_iq`: mother's IQ
* `mom_age`: mother's age
"""

# ╔═╡ 2fda1cb5-ccc0-41e2-82a6-7549ffa082aa
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/kidiq.csv"

# ╔═╡ b2fa0a26-177d-4cb1-a8e6-73d6becf07e1
kidiq = CSV.read(download(url), DataFrame)

# ╔═╡ dcfc7263-5a89-4632-951d-591ddae5f447
md"""
Using `kid_score` as dependent variable and `mom_hs` along with `mom_iq` as independent variables with a moderation (interaction) effect:
"""

# ╔═╡ 22b266c4-a92b-4f4c-b23c-2fa3cf3a2afb
fm = @formula(kid_score ~ mom_hs * mom_iq)

# ╔═╡ 4f9ff3fd-3f04-49a5-924e-aa23702e75a0
md"""
We instantiate our model with `turing_model` passing a third argument `Student()` to
indicate that the model is a robust regression with the Student's t-distribution:
"""

# ╔═╡ 3f4241d4-1c76-4d7c-99c1-aaa2111385f9
model = turing_model(fm, kidiq, Student());

# ╔═╡ 7bbe4fe4-bcaf-4699-88e4-0dff92250d30
chn = sample(model, NUTS(), 2_000);

# ╔═╡ c2c17f3b-8728-4b0c-ab06-639862ca31f9
describe(chn)[1]

# ╔═╡ a90f1d84-b39c-47b5-a251-3a6f0dfae4b3
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═cea125d8-7303-11ec-3f43-6b79e00bca6a
# ╠═cd2dc745-d2bf-4960-854d-bad17cf767ab
# ╠═e0b79bde-f1c1-4256-866e-eb0649e77cb7
# ╠═5a7e2bbe-ded0-4ce4-b24c-6b4387a7e80d
# ╠═2fda1cb5-ccc0-41e2-82a6-7549ffa082aa
# ╠═b2fa0a26-177d-4cb1-a8e6-73d6becf07e1
# ╠═6002455d-45ba-42f4-9601-03754c2906d1
# ╠═dcfc7263-5a89-4632-951d-591ddae5f447
# ╠═22b266c4-a92b-4f4c-b23c-2fa3cf3a2afb
# ╠═4f9ff3fd-3f04-49a5-924e-aa23702e75a0
# ╠═3f4241d4-1c76-4d7c-99c1-aaa2111385f9
# ╠═7bbe4fe4-bcaf-4699-88e4-0dff92250d30
# ╠═c2c17f3b-8728-4b0c-ab06-639862ca31f9
# ╠═a90f1d84-b39c-47b5-a251-3a6f0dfae4b3
