# Tiny results wrapper to dispatch plots on.
struct PixelFlippingResult{T}
    MIF::Vector{T}
    LIF::Vector{T}
    occl::Vector{T}
end

mif(res::PixelFlippingResult) = res.MIF
lif(res::PixelFlippingResult) = res.LIF
occlusion(res::PixelFlippingResult) = res.occl
srg(res::PixelFlippingResult) = sum(max.(0, res.LIF - res.MIF)) / steps(res) # integrate area between MIF and LIF
steps(res::PixelFlippingResult) = length(res.MIF) - 1

# Plot pixel-flipping curves in terminal
function unicode_plot(res::PixelFlippingResult)
    MIF = mif(res)
    LIF = lif(res)
    occl = occlusion(res)
    plt = lineplot(
        occl,
        MIF;
        title="Pixel flipping curve (SRG=$(srg(res)))",
        name="MIF",
        xlabel="Occlusion (%)",
        ylabel="p",
    )
    lineplot!(plt, occl, LIF; color=:cyan, name="LIF")
    return plt
end
