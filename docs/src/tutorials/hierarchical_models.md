# Hierarchical Models

TuringGLM currently only supports hierarchical models with a single random-intercept.
This is done by using the `(1 | group)` inside the `@formula` macro.

For our **Hierarchical Model** example, let's use a famous dataset called `cheese` (Boatwright, McCulloch & Rossi, 1999), which is data from cheese ratings.
A group of 10 rural and 10 urban raters rated 4 types of different cheeses (A, B, C and D) in two samples.
So we have $4 \cdot 20 \cdot 2 = 160$ observations and 4 variables:

* `cheese`: type of cheese from `A` to `D`
* `rater`: id of the rater from `1` to `10`
* `background`: type of rater, either `rural` or `urban`
* `y`: rating of the cheese

```@repl
# Load Packages
using CSV
using DataFrame

# Import the dataset as a DataFrame with CSV.jl
cheese = CSV.read("https://github.com/TuringLang/TuringGLM.jl/raw/main/data/cheese.csv", DataFrame)

# Load TuringGLM
using TuringGLM

# Using y as dependent variable and background as independent variable
# with a varying-intercept per cheese type
f = @formula(y ~ (1 | cheese) + background)

# Now we instantiate our model with turing_model without specifying any model,
# thus the default model will be used: Gaussian()
m = turing_model(f, cheese)

# Sample the model using NUTS
chn = sample(m, NUTS(), 2_000)
```

## References

Boatwright, P., McCulloch, R., & Rossi, P. (1999). Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model. Journal of the American Statistical Association, 94(448), 1063–1073.
