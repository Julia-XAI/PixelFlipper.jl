using PixelFlipper
using ExplainableAI
using VisionHeatmaps         # visualization of explanations as heatmaps
using Zygote                 # load autodiff backend for gradient-based methods
using Flux, Metalhead        # pre-trained vision models in Flux
using DataAugmentation       # input preprocessing
using HTTP, FileIO, ImageIO  # load image from URL
using ImageInTerminal        # show heatmap in terminal

# Load & prepare model
model = VGG(16, pretrain = true)

# Load input
url = HTTP.URI(
    "https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg",
)
img = load(url)

# Preprocess input
mean = (0.485f0, 0.456f0, 0.406f0)
std = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(mean, std)
input = apply(tfm, Image(img))
input = reshape(input.data, 224, 224, 3, :)

# Run XAI method
analyzer = Gradient(model)
expl = analyze(input, analyzer)
heatmap(expl)

## Create imputer
imputer = ConstantImputer()
