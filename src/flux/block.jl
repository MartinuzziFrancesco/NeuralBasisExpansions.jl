struct NBeatsBlock{F}
    fc_stack::Chain
    fc_b::Dense
    fc_f::Dense
    basis_layer_b::F #AbstractBasisLayer
    basis_layer_f::F #AbstractBasisLayer
end

function NBeatsBlock(
    input_size::Int,
    theta_size::Int,
    basis_function,
    layer_size::Int,
    share_thetas::Bool=false,
    num_layers::Int = 4
)
    layer_sequence = [Dense(input_size, layer_size, relu)]
    append!(layer_sequence, [Dense(layer_size, layer_size, relu) for _ in 2:num_layers])
    layers_chain = Chain(layer_sequence...)
    fc_b = Dense(layer_size, theta_size)
    fc_f = share_thetas ? fc_b : Dense(layer_size, theta_size)
    basis_layer_b = Basis()

    return NBeatsBlock(
        layers_chain,
        fc_b, #bias = false
        fc_f, #bias = false
        basis_function)
end

function (block::NBeatsBlock)(x::AbstractArray)
    block_input = block.fc_stack(x)
    basis_parameters = block.basis_parameters(block_input)
    return block.basis_function(basis_parameters)
end

struct 