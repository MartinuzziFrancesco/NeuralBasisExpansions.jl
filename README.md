# NBeats

[![Build Status](https://github.com/MartinuzziFrancesco/NBeats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MartinuzziFrancesco/NBeats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MartinuzziFrancesco/NBeats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinuzziFrancesco/NBeats.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Full sin example with helper functions is given in the `example` folder, under `readme.jl`.
```julia
# Model parameters
forecast_length = 5
backcast_length = 2*forecast_length
batch_size = 32
hidden_units = 128
theta_dims = (4, 8)
blocks_per_stack = 3

# Generate and batch the data
data = generate_sine_data(1000, backcast_length, forecast_length)
train_data, test_data = data[1:800], data[801:end]
train_batches = batch_data(train_data, batch_size)
test_batches = batch_data(test_data, batch_size)

# Create the NBeatsNet model
model = NBeatsNet(
    stacks=[generic_basis, trend_basis],
    blocks_stacks=blocks_per_stack,
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    thetas_dim=theta_dims,
    hidden_units=hidden_units
)

# Loss function and optimizer
loss_fn(x, y) = Flux.mse(model(x)[2], y)
optimizer = Flux.ADAM(0.001)

# Training loop
epochs = 50
for epoch in 1:epochs
    Flux.train!(loss_fn, Flux.params(model), train_batches, optimizer)
    train_loss = mean([loss_fn(getindex(batch, 1), getindex(batch, 2)) for batch in train_batches])
    test_loss = mean([loss_fn(getindex(batch, 1), getindex(batch, 2)) for batch in test_batches])
    println("Epoch $epoch: Train Loss = $train_loss, Test Loss = $test_loss")
end

# Forecast using the model (example)
x_test, y_true = test_batches[1]
y_pred = model(x_test)[2]

mse, mae, r_squared = evaluate_predictions(y_true, y_pred)

println("Mean Squared Error: $mse")
println("Mean Absolute Error: $mae")
println("R-squared: $r_squared")
```

Quick example with random data to test the model
```julia
forecast_length = 5
backcast_length = 10
blocks_stacks = 3
thetas_dim = (4, 8)
hidden_units = 256

nbeats_net = NBeatsNet(
    stacks = [trend_basis, seasonality_basis],
    blocks_stacks = blocks_stacks,
    forecast_length = forecast_length,
    backcast_length = backcast_length,
    thetas_dim = thetas_dim,
    share_weights = false,
    hidden_units = hidden_units
)

# Create a batch of input data
batch_size = 3  # Number of instances in the batch
input_data = randn(Float32, backcast_length, batch_size)

backcast_output, forecast_output = nbeats_net(input_data)
```
