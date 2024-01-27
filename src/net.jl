struct NBeatsNet
    stacks::Vector{Vector{NBeatsBlock}}
    forecast_length::Int
    backcast_length::Int
end

function NBeatsNet(;
    stacks = [trend_basis, seasonality_basis],
    blocks_stacks::Int = 3,
    forecast_length::Int=5,
    backcast_length::Int=10,
    thetas_dim = (4, 8),
    share_weights::Bool=false,
    hidden_units = 256
)
    net_stacks = Vector{Vector{NBeatsBlock}}()

    for (i, basis_function) in enumerate(stacks)
        theta_size = thetas_dim[i]  # Select theta_size from thetas_dim tuple
        stack = [NBeatsBlock(
            theta_size,
            hidden_units,
            basis_function;
            share_thetas=share_weights,
            num_layers=4,
            backcast_length=backcast_length,
            forecast_length=forecast_length
        ) for _ in 1:blocks_stacks]
        push!(net_stacks, stack)
    end

    return NBeatsNet(net_stacks, forecast_length, backcast_length)
end

function (net::NBeatsNet)(x::AbstractArray)
    backcast = x
    forecast = zeros(eltype(x), net.forecast_length, size(backcast, 2))

    for stack in net.stacks
        for block in stack
            block_backcast, block_forecast = block(backcast)
            backcast -= block_backcast
            forecast += block_forecast
        end
    end

    return backcast, forecast
end

function split(arr, size)
    arrays = []
    while length(arr) > size
        slice_ = arr[1:size]
        push!(arrays, slice_)
        arr = arr[(size+1):end]
    end
    push!(arrays, arr)
    return arrays
end

function train!(model::NBeatsNet,
    x_train,
    y_train;
    optimizer=Flux.ADAM(0.001),
    loss_fn=Flux.mse,
    epochs=10,
    batch_size=32,
    validation_data = nothing
)
    for epoch in 1:epochs
        # Split the training data into batches
        x_train_batches = split(x_train, batch_size)
        y_train_batches = split(y_train, batch_size)

        # Shuffle the indices of the batches
        shuffled_indices = shuffle(1:length(x_train_batches))

        total_loss = 0.0
        for batch_id in shuffled_indices
            batch_x = x_train_batches[batch_id]
            batch_y = y_train_batches[batch_id]

            # Gradient calculation and parameter update
            grads = gradient(() -> loss_fn(model(batch_x)[2], batch_y), params(model))
            Flux.Optimise.update!(optimizer, params(model), grads)

            # Update loss
            total_loss += loss_fn(model(batch_x)[2], batch_y)
        end

        # Calculate average training loss
        avg_train_loss = total_loss / length(shuffled_indices)

        # Validation (if provided)
        val_loss = "undefined"
        if validation_data !== nothing
            x_val, y_val = validation_data
            val_loss = loss_fn(model(x_val)[2], y_val)
        end

        # Print epoch results
        println("Epoch $epoch/$epochs - loss: $avg_train_loss - val_loss: $val_loss")
    end
end

function predict(model::NBeatsNet, x, return_backcast=false)
    backcast, forecast = model(x)

    # Handle 3D inputs
    if ndims(x) == 3
        backcast = reshape(backcast, size(backcast, 1), size(backcast, 2), 1)
        forecast = reshape(forecast, size(forecast, 1), size(forecast, 2), 1)
    end

    # Optionally return backcast
    if return_backcast
        return backcast
    else
        return forecast
    end
end
