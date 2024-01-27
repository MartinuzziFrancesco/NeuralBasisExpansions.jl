struct NBeatsNet
    blocks::Vector{<:Any}
end

function NBeatsNet(;
    stacks = [trend_basis, seasonality_basis],
    blocks_stacks::Int = 3,
    forecast_length::Int=5,
    backcast_length::Int=10,
    thetas_dim = (4, 8),
    share_weights::Bool=false,
    hidden_units = 256,
    num_harmonics = nothing
)

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
