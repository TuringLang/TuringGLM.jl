### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ bfc8e740-7302-11ec-19e2-8995cf677fbf
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
    Pkg.develop(; path=pkg_dir)
    Pkg.instantiate()
end

# ╔═╡ 423d0cd9-010f-43b2-a39e-8f8f5c0233a8
using CSV

# ╔═╡ a78a82f2-2ee5-41a5-a0e3-bae1639e3859
using DataFrames

# ╔═╡ 0f6e5024-402e-47ba-ab6c-bd75b5b5bfb0
using TuringGLM

# ╔═╡ 9ae04dd9-c612-4e88-bb02-38d0f325bfe9
md"""
For our example on **Poisson Regression**, let's use a famous dataset called `roaches` (Gelman & Hill, 2007), which is data on the efficacy of a
pest management system at reducing the number of roaches in urban apartments.
It has 262 observations and the following variables:

* `y` -- number of roaches caught.
* `roach1` -- pretreatment number of roaches.
* `treatment` -- binary/dummy (0 or 1) for treatment indicator.
* `senior` -- binary/dummy (0 or 1) for only elderly residents in building.
* `exposure2` -- number of days for which the roach traps were used
"""

# ╔═╡ 15670545-e2c5-4396-89d8-dddd919925d9
url = "https://github.com/TuringLang/TuringGLM.jl/raw/main/data/roaches.csv";

# ╔═╡ 84b8c1ec-779c-4113-ae6a-084955eba852
roaches = CSV.read(download(url), DataFrame)

# ╔═╡ ca606aee-3115-4e43-97ca-4aae60a387bf
md"""
Using `y` as dependent variable and `roach1`, `treatment`, and `senior` as independent variables:
"""

# ╔═╡ 9f88708a-7825-4c96-a720-ac7bf25e2591
fm = @formula(y ~ roach1 + treatment + senior)

# ╔═╡ 65f4f379-e9a5-4571-bc80-77c024f3f560
md"""
We instantiate our model with `turing_model` passing a third argument `Pois()` to
indicate that the model is a Poisson Regression
"""

# ╔═╡ 9147ed9e-a047-42ca-aa36-f522fad8388b
model = turing_model(fm, roaches, Pois());

# ╔═╡ aeeac527-2c32-407f-84a1-912cc74f51b7
md"""
Sample the model using the `NUTS` sampler and 2,000 samples:
"""

# ╔═╡ abe86f76-2d0f-42e5-8faa-52aa3ddea126
chn = sample(model, NUTS(), 2_000);

# ╔═╡ d322ced4-9334-4a15-b3f9-e7b05a8a7d52
describe(chn)[1]

# ╔═╡ 71fe7834-117d-4d52-a86d-25d4a52bc9a0
md"""
## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
"""

# ╔═╡ Cell order:
# ╠═bfc8e740-7302-11ec-19e2-8995cf677fbf
# ╠═9ae04dd9-c612-4e88-bb02-38d0f325bfe9
# ╠═423d0cd9-010f-43b2-a39e-8f8f5c0233a8
# ╠═a78a82f2-2ee5-41a5-a0e3-bae1639e3859
# ╠═15670545-e2c5-4396-89d8-dddd919925d9
# ╠═84b8c1ec-779c-4113-ae6a-084955eba852
# ╠═0f6e5024-402e-47ba-ab6c-bd75b5b5bfb0
# ╠═ca606aee-3115-4e43-97ca-4aae60a387bf
# ╠═9f88708a-7825-4c96-a720-ac7bf25e2591
# ╠═65f4f379-e9a5-4571-bc80-77c024f3f560
# ╠═9147ed9e-a047-42ca-aa36-f522fad8388b
# ╠═aeeac527-2c32-407f-84a1-912cc74f51b7
# ╠═abe86f76-2d0f-42e5-8faa-52aa3ddea126
# ╠═d322ced4-9334-4a15-b3f9-e7b05a8a7d52
# ╠═71fe7834-117d-4d52-a86d-25d4a52bc9a0
