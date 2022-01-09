# Poisson Regression

For our example on **Poisson Regression**, let's use a famous dataset called `roaches` (Gelman & Hill, 2007), which is data on the efficacy of a
pest management system at reducing the number of roaches in urban apartments.
It has 262 observations and the following variables:

* `y` -- number of roaches caught.
* `roach1` -- pretreatment number of roaches.
* `treatment` -- binary/dummy (0 or 1) for treatment indicator.
* `senior` -- binary/dummy (0 or 1) for only elderly residents in building.
* `exposure2` -- number of days for which the roach traps were used

```@repl
# Load Packages
using CSV
using DataFrame

# Import the dataset as a DataFrame with CSV.jl
roaches = CSV.read("https://github.com/TuringLang/TuringGLM.jl/raw/main/data/roaches.csv", DataFrame)

# Load TuringGLM
using TuringGLM

# Using y as dependent variable and roach1, treatment, and senior as independent variables
f = @formula(y ~ roach1 + treatment + senior)

# Now we instantiate our model with turing_model passing a third argument Pois() to
# indicate that the model is a Poisson Regression
m = turing_model(f, roaches, Pois())

# Sample the model using NUTS
chn = sample(m, NUTS(), 2_000)
```

## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
