"""
    run(pixelflipping, model, input, explanation::Explanation)
    run(pixelflipping, model, input, values::AbstractArray) 
    run(pixelflipping, model, input, analyzer::AbstractXAIMethod)

Run the `PixelFlipping` method on the given model, input and explanation.
"""
function run(pf::PixelFlipping, model, input::AbstractWHCN{T}, x::AbstractWHCN) where {T}
    selection = select(x, pf.selector)
    n, batchsize = size(selection)

    # Allocate outputs
    MIF = Vector{T}(undef, pf.steps + 1) # Most influential first
    LIF = Vector{T}(undef, pf.steps + 1) # Least influential first

    # Run initial forward pass
    output = model(input)
    output_selection = pf.output_selector(output)

    # Compute mean probability on activated neurons over batch
    pmean = mean_probability(output, output_selection)
    MIF[1] = pmean
    LIF[1] = pmean

    ## Compute MIF curve
    input_mif = deepcopy(input)
    npart = ceil(Int, n / pf.steps) # length of a partition
    for (i, range) in Iterators.enumerate(Iterators.partition(1:n, npart))
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
    for (i, range) in Iterators.enumerate(Iterators.partition(n:-1:1, npart))
        # Modify input in-place, iterating over multiple rows of selection at a time
        for CI in selection[range, :]
            impute!(input_lif, CI, pf.imputer)
        end

        # Run new forward pass
        output = model(input_lif)
        LIF[i + 1] = mean_probability(output, output_selection)
    end

    return PixelFlippingResult(MIF, LIF)
end

function mean_probability(output, output_selection)
    ps = softmax(output)[output_selection]
    return mean(ps)
end

# Convenient ways to call PixelFlipping using XAIBase API
function run(pf::PixelFlipping, model, input::AbstractWHCN, expl::Explanation)
    return run(pf, model, input, expl.val)
end

function run(pf::PixelFlipping, model, input::AbstractWHCN, analyzer::AbstractXAIMethod;)
    expl = analyze(input, analyzer)
    return run(pf, model, input, expl)
end
