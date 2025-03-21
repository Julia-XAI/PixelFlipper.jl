"""
    evaluate(pixelflipping, model, input, explanation::Explanation)
    evaluate(pixelflipping, model, input, values::AbstractArray) 
    evaluate(pixelflipping, model, input, analyzer::AbstractXAIMethod)

Run the `PixelFlipping` method on the given model, input and explanation.
"""
function evaluate(
    pf::PixelFlipping, model, input::AbstractWHCN{T}, x::AbstractWHCN
) where {T}
    selection = select(x, pf.selector)
    n, batchsize = size(selection)

    # Allocate outputs
    MIF = Vector{T}(undef, pf.steps + 1) # Most influential first
    LIF = Vector{T}(undef, pf.steps + 1) # Least influential first
    occl = collect(range(0.0f0, 1.0f0; length=pf.steps + 1))

    # Run initial forward pass
    output = model(input)
    output_selection = pf.output_selector(output)

    # Compute mean probability on activated neurons over batch
    pmean = mean_probability(output, output_selection)
    MIF[1] = pmean
    LIF[1] = pmean

    ## Compute MIF curve
    npart = ceil(Int, n / pf.steps) # length of a partition
    input_mif = deepcopy(input)
    @showprogress desc = "Computing MIF curve..." for (i, range) in Iterators.enumerate(
        Iterators.partition(1:n, npart)
    )
        # Modify input in-place, iterating over multiple rows of selection at a time
        for CI in selection[range, :]
            impute!(input_mif, CI, pf.imputer)
        end

        # Run new forward pass
        output = model(input_mif)
        MIF[i + 1] = mean_probability(output, output_selection)
    end

    ## Compute LIF curve
    input_lif = deepcopy(input)
    @showprogress desc = "Computing LIF curve..." for (i, range) in Iterators.enumerate(
        Iterators.partition(n:-1:1, npart)
    )
        # Modify input in-place, iterating over multiple rows of selection at a time
        for CI in selection[range, :]
            impute!(input_lif, CI, pf.imputer)
        end

        # Run new forward pass
        output = model(input_lif)
        LIF[i + 1] = mean_probability(output, output_selection)
    end

    return PixelFlippingResult(MIF, LIF, occl)
end

function mean_probability(output, output_selection)
    ps = softmax(output)[output_selection]
    return mean(ps)
end

# Convenient ways to call PixelFlipping using XAIBase API
function evaluate(pf::PixelFlipping, model, input::AbstractWHCN, expl::Explanation)
    return evaluate(pf, model, input, expl.val)
end

function evaluate(
    pf::PixelFlipping, model, input::AbstractWHCN, analyzer::AbstractXAIMethod;
)
    expl = analyze(input, analyzer)
    return evaluate(pf, model, input, expl)
end
