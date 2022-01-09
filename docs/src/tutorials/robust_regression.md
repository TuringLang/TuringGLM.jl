# Robust Regression

For the **Robust Regression** with Student-$t$ distribution as the likelihood, we'll use a famous dataset called `kidiq` (Gelman & Hill, 2007), which is data from a survey of adult American women and their respective children.
Dated from 2007, it has 434 observations and 4 variables:

* `kid_score`: child's IQ
* `mom_hs`: binary/dummy (0 or 1) if the child's mother has a high school diploma
* `mom_iq`: mother's IQ
* `mom_age`: mother's age

```@repl
# Load Packages
using CSV
using DataFrame

# Import the dataset as a DataFrame with CSV.jl
kidiq = CSV.read("https://github.com/TuringLang/TuringGLM.jl/raw/main/data/kidiq.csv", DataFrame)

# Load TuringGLM
using TuringGLM

# Using kid_score as dependent variable and mom_hs along with mom_iq as independent
# variables with a moderation (interaction) effect
f = @formula(kid_score ~ mom_hs * mom_iq)

# Now we instantiate our model with turing_model passing a third argument Student() to
# indicate that the model is a Robust Regression with Student-t distribution
m = turing_model(f, kidiq, Student())

# Sample the model using NUTS
chn = sample(m, NUTS(), 2_000)
```

## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
