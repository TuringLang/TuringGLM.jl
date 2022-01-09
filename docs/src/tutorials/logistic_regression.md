# Logistic Regression

For our tutorial on **Logistic Regression**, let's use a famous dataset called `wells` (Gelman & Hill, 2007),
which is data from a survey of 3,200 residents in a small area of Bangladesh suffering from arsenic contamination of groundwater.
Respondents with elevated arsenic levels in their wells had been encouraged to switch their water source to a safe public or private well in the nearby area and the survey was conducted several years later to learn which of the affected residents had switched wells.
It has 3,200 observations and the following variables:

* `switch` -- binary/dummy (0 or 1) for well-switching.
* `arsenic` -- arsenic level in respondent's well.
* `dist` -- distance (meters) from the respondent's house to the nearest well with safe drinking water.
* `association` -- binary/dummy (0 or 1) if member(s) of household participate in community organizations.
* `educ` -- years of education (head of household).

```@repl
# Load Packages
using CSV
using DataFrame

# Import the dataset as a DataFrame with CSV.jl
wells = CSV.read("https://github.com/TuringLang/TuringGLM.jl/raw/main/data/wells.csv", DataFrame)

# Load TuringGLM
using TuringGLM

# Using switch as dependent variable and dist, arsenic, assoc, and educ as independent variables
f = @formula(switch ~ dist + arsenic + assoc + educ)

# Now we instantiate our model with turing_model passing a third argument Logistic() to
# indicate that the model is a Logistic Regression
m = turing_model(f, wells, Logistic())

# Sample the model using NUTS
chn = sample(m, NUTS(), 2_000)
```

## References

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
