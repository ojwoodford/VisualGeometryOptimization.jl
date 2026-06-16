import NLLSsolver: NLLSProblem

# Compute the Area Under Curve for errors, truncated at a given threshold
function computeauc(errors::Vector, threshold)
    # Add the origin and scale the errors by the threshold
    nerrors = length(errors)
    errors = vcat(0.0, abs.(errors) .* (1.0 / threshold))

    # Sort the errors and truncate, compute recall
    sort!(errors)
    if errors[end] > 1.0
        cutoff = findfirst(x -> x > 1.0, errors)
        recall = (cutoff - 1.0) / nerrors
        recallfinal = (1.0 - errors[cutoff-1]) / ((errors[cutoff] - errors[cutoff-1]) * nerrors)
        errors[cutoff] = 1.0
        resize!(errors, cutoff)
        recall = vcat(LinRange(0.0, recall, cutoff-1), recall + recallfinal)
    else
        recall = vcat(LinRange(0.0, 1.0, nerrors+1), 1.0)
        errors = vcat(errors, 1.0)
    end

    # Compute the AUC
    return 0.5 * sum(diff(errors) .* (recall[1:end-1] .+ recall[2:end]))
end

function computeerrors(residuals, variables)
    # Compute all the errors
    errors = Vector{Float64}(undef, length(residuals))
    for (ind, res) in enumerate(residuals)
        errors[ind] = norm(NLLSsolver.computeresidual(res, NLLSsolver.getvars(res, variables)...))
    end
    return errors
end

computeauc(problem::NLLSProblem, threshold, restype::DataType=first(keys(problem.costs.data))) = computeauc(computeerrors(problem.costs.data[restype], problem.variables), threshold)
