using Optim, ForwardDiff

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [0.0, 0.0]

# gradient free optimizers
optimize(f, x0, NelderMead())

# gradient based optimizers
function g(x)
    return ForwardDiff.gradient(f, x)
end
optimize(f, g, x0, LBFGS(); inplace = false)