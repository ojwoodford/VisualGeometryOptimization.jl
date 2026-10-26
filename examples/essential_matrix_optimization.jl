# Essential matrix optimization on the CVPR 2020 RANSAC tutorial data.
using VisualGeometryOptimization, NLLSsolver

problem = loadEssentialMatrixproblem("st_peters_square", 1)
options = NLLSOptions(; reldcost=1.0e-6, maxiters=100)
NLLSsolver.robustkernel(::VisualGeometryOptimization.EpipolarResidualPair) = Huber2oKernel(2.)

function myfun(problem, options)
    result = optimize!(problem, options, nothing, printoutcallback)
    display(result)
    return nothing
end

myfun(problem, options)
