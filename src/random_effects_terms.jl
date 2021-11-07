# TODO: MixedModels.jl has a lot of dependencies and we just want the random effects terms
# for the @formula parsing. This means that we need to build from scratch:
#
# 1. rhs parsing for random-intercept (1 | group)
# 2. rhs parsing for random slope (x1 | group)
# 3. rhs parsing for random-intercept-slope (1 + x1 | group)
#
# Caution for stuff with/without fixed-effects:
#   y ~ 1 + (1 | group)
#   y ~ 0 + (1 | group)
#   y ~ x1 + (1 + x1| group)
#   y ~ x1 + (0 + x1| group)
#   y ~ 0 + x1 + (1 + x1| group)

# TODO: zerocorr function for random-effects
