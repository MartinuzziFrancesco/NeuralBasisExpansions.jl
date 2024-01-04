struct NBeatsBlock{F}
    layers::Chain
    basis_parameters::Dense
    basis_function::F
end

function NBeatsBlock(
    input_size::Int, theta_size::Int, basis_function, layers::Int, layer_size::Int
)
    layer_sequence = [Dense(input_size, layer_size, relu)]
    append!(layer_sequence, [Dense(layer_size, layer_size, relu) for _ in 2:layers])
    layers_chain = Chain(layer_sequence...)
    basis_parameters_layer = Dense(layer_size, theta_size)

    return NBeatsBlock(layers_chain, basis_parameters_layer, basis_function)
end

function (block::NBeatsBlock)(x::AbstractArray)
    block_input = block.layers(x)
    basis_parameters = block.basis_parameters(block_input)
    return block.basis_function(basis_parameters)
end
