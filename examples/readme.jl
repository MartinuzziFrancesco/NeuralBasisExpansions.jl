using Pkg: Pkg
__DIR = @__DIR__
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()

using NBeats
using Flux
using Statistics
using Plots

include("data_utils.jl")

# Generate a simple sine wave dataset
function generate_sine_data(num_points, backcast_length, forecast_length)
    x = Float32.(sin.(range(0; stop=4π, length=num_points)))
    data = [
        (
            x[i:(i + backcast_length - 1)],
            x[(i + backcast_length):(i + backcast_length + forecast_length - 1)],
        ) for i in 1:(num_points - backcast_length - forecast_length)
    ]
    return data
end

# Model parameters
forecast_length = 5
backcast_length = 2 * forecast_length
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
model = NBeatsNet(;
    stacks=[generic_basis, trend_basis],
    blocks_stacks=blocks_per_stack,
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    thetas_dim=theta_dims,
    hidden_units=hidden_units,
)

# Loss function and optimizer
loss_fn(x, y) = Flux.mse(model(x)[2], y)
optimizer = Flux.ADAM(0.001)

# Training loop
epochs = 50
for epoch in 1:epochs
    Flux.train!(loss_fn, Flux.params(model), train_batches, optimizer)
    train_loss = mean([
        loss_fn(getindex(batch, 1), getindex(batch, 2)) for batch in train_batches
    ])
    test_loss = mean([
        loss_fn(getindex(batch, 1), getindex(batch, 2)) for batch in test_batches
    ])
    println("Epoch $epoch: Train Loss = $train_loss, Test Loss = $test_loss")
end

# Forecast using the model (example)
x_test, y_true = test_batches[1]
y_pred = model(x_test)[2]

mse, mae, r_squared = evaluate_predictions(y_true, y_pred)

println("Mean Squared Error: $mse")
println("Mean Absolute Error: $mae")
println("R-squared: $r_squared")

plot(y_true[:, end]; label="True")
plot!(y_pred[:, end]; label="Predicted")

# Modify generate_sine_data if necessary to ensure data is in column vector format
function generate_sine_data(num_points, backcast_length, forecast_length)
    x = sin.(range(0; stop=4π, length=num_points))
    data = [
        (
            x[i:(i + backcast_length - 1)]',
            x[(i + backcast_length):(i + backcast_length + forecast_length - 1)]',
        ) for i in 1:(num_points - backcast_length - forecast_length)
    ]
    return data
end

# Generate data
data = generate_sine_data(1000, backcast_length, forecast_length)

# Ensure each instance is a column vector
println("Backcast shape: ", size(first(data)[1]))
println("Forecast shape: ", size(first(data)[2]))

# Create the NBeatsNet model
model = NBeatsNet(;
    stacks=[generic_basis, trend_basis],
    blocks_stacks=blocks_per_stack,
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    thetas_dim=theta_dims,
    hidden_units=hidden_units,
)

# Training loop with custom train! function
train!(
    model,
    x_train,
    y_train;
    optimizer=Flux.ADAM(0.001),
    loss_fn=Flux.mse,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
)
