using VisualGeometryOptimization, NLLSsolver

NLLSsolver.robustkernel(::VisualGeometryOptimization.BALResidual) = Huber2oKernel(2.)
optimizeBALproblem("problem-16-22106");
