# Compute the Area Under Curve for errors, truncated at a given threshold
function computeauc(problem, threshold, residuals)
    # Compute all the errors
    invthreshold = 1.0 / threshold
    errors = Vector{Float64}(undef, length(residuals)+1)
    ind = 1
    errors[ind] = 0.0
    for res in residuals
        ind += 1
        errors[ind] = norm(NLLSsolver.computeresidual(res, NLLSsolver.getvars(res, problem.variables)...)) * invthreshold
    end

    # Sort the errors and truncate, compute recall
    sort!(errors)
    if errors[end] > 1.0
        cutoff = findfirst(x -> x > 1.0, errors)
        recall = (cutoff - 1.0) / length(residuals)
        recallfinal = (1.0 - errors[cutoff-1]) / ((errors[cutoff] - errors[cutoff-1]) * length(residuals))
        errors[cutoff] = 1.0
        resize!(errors, cutoff)
        recall = vcat(range(0.0, recall, cutoff-1), recall + recallfinal)
    else
        recall = vcat(range(0.0, 1.0, length(errors)), 1.0)
        errors = vcat(errors, 1.0)
    end

    # Compute the AUC
    return 0.5 * sum(diff(errors) .* (recall[1:end-1] .+ recall[2:end]))
end