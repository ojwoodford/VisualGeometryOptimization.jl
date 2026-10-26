using VisualGeometryOptimization, NLLSsolver

startproblem = loadBALproblem("problem-21-11315")
options = NLLSOptions(; reldcost=1.0e-6, maxiters=1000)

NLLSsolver.robustkernel(::BALResidual) = GemanMcclureKernel(3.)
problem = deepcopy(startproblem)
result = optimize!(problem, options)
niters = result.niterations
NLLSsolver.robustkernel(::BALResidual) = GemanMcclureKernel(1.)
result = optimize!(problem, options)
niters += result.niterations
auc1 = computeauc(problem, 2.0, BALResidual{Float64})

mutable struct ContaminatedGaussianReference
    kernel::ContaminatedGaussian
end
mycontamref = ContaminatedGaussianReference(ContaminatedGaussian(1.0, 100.0, 0.1))

function optimisekernel!(problem, vars=problem.variables)
    # Compute the squared errors
    squarederrors = [NLLSsolver.sqnorm(NLLSsolver.computeresidual(res, getvars(res, vars)...)) for res in problem.costs.data[BALResidual{Float64}]]
    # Optimize the kernel parameters
    mycontamref.kernel = NLLSsolver.optimize(mycontamref.kernel, squarederrors)
    return nothing
end

function emcallback(cost, problem, data, trailingargs...)
    if mod(data.iternum, 5) != 4
        return cost, 0
    end
    # Optimize the kernel parameters
    optimisekernel!(problem, problem.varnext)
    # Recompute the cost
    data.timecost += NLLSsolver.@elapsed_ns newcost = NLLSsolver.cost(problem.varnext, problem.costs)
    data.costcomputations += 1
    # Return cost and do not trigger termination
    return newcost, 0
end

NLLSsolver.robustkernel(::BALResidual) = mycontamref.kernel
problem = deepcopy(startproblem)
result = optimize!(problem, options, nothing, emcallback)
niters2 = result.niterations
auc2 = computeauc(problem, 2.0, BALResidual{Float64})

show((niters, niters2, auc1, auc2))
