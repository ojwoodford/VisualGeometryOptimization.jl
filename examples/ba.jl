using VisualGeometryOptimization, NLLSsolver, Static

problem = loadBALproblem("problem-16-22106")
options = NLLSOptions(; reldcost=1.0e-6, maxiters=100, iterator=:levenbergmarquardt, numthreads=static(1))
NLLSsolver.robustkernel(::VisualGeometryOptimization.BALResidual) = HuberKernel(3.)

function myfun(problem, options)
    result = optimize!(problem, options, nothing, printoutcallback)
    display(result)
    return nothing
end

myfun(problem, options)
