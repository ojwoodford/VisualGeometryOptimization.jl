# Essential matrix optimization on the CVPR 2020 RANSAC tutorial data.
using VisualGeometryOptimization, NLLSsolver

NLLSsolver.robustkernel(::VisualGeometryOptimization.EpipolarResidualPair) = GemanMcclure2oKernel(1.)
optimizeEssentialMatrixproblem("st_peters_square", 1);
