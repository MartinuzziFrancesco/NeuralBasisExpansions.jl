struct NBeatsBlock{F}
    fc_stack::Chain
    basis_layer::F #AbstractBasisLayer
end

function NBeatsBlock(
    theta_size::Int,
    layer_size::Int,
    basis_function;
    share_thetas::Bool=false,
    num_layers::Int = 4,
    backcast_length::Int=10,
    forecast_length::Int=5
)
    layer_sequence = [Dense(backcast_length, layer_size, relu)]
    append!(layer_sequence, [Dense(layer_size, layer_size, relu) for _ in 2:num_layers])
    layers_chain = Chain(layer_sequence...)
    basis_layer = BasisLayer(
        layer_size, theta_size, basis_function;
        backcast_length = backcast_length,
        forecast_length = forecast_length,
        share_thetas = share_thetas
    )

    return NBeatsBlock(
        layers_chain,
        basis_layer)
end

Flux.@functor NBeatsBlock
Flux.trainable(block::NBeatsBlock) = (block.fc_stack, block.basis_layer)

function (block::NBeatsBlock)(x::AbstractArray)
    block_input = block.fc_stack(x)
    backcast, forecast = block.basis_layer(block_input)
    return backcast, forecast
end

