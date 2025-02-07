using PixelFlipper
using Documenter

DocMeta.setdocmeta!(PixelFlipper, :DocTestSetup, :(using PixelFlipper); recursive=true)

makedocs(;
    modules=[PixelFlipper],
    authors="Adrian Hill <gh@adrianhill.de>",
    sitename="PixelFlipper.jl",
    format=Documenter.HTML(;
        canonical="https://adrhill.github.io/PixelFlipper.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/adrhill/PixelFlipper.jl",
    devbranch="main",
)
