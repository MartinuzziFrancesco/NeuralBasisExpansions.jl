module NBeats

using Flux
using LinearAlgebra

abstract type AbstractBlock end

export NBeatsBlock, NBeatsNet
export GenericBasis, TrendBasis, SeasonalityBasis

struct NBeatsNet
    blocks::Vector{<:Any}
end

function (model::NBeatsNet)(x::AbstractArray, input_mask::AbstractArray)
    residuals = reverse(x; dims=2)
    input_mask = reverse(input_mask; dims=2)
    forecast = x[:, end:end]

    for block in model.blocks
        backcast, block_forecast = block(residuals)
        residuals = (residuals - backcast) .* input_mask
        forecast = forecast + block_forecast
    end

    return forecast
end

include("block.jl")
include("basis.jl")

end
