using Pkg: Pkg
__DIR = @__DIR__
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()

using Downloads
using CSV
using DataFrames
using NeuralBasisExpansions

include("data_utils.jl")
download_m4()

forecast_length = 5
backcast_length = 2 * forecast_length
batch_size = 32

train_batches, test_batches = get_m4_data(backcast_length, forecast_length, batch_size)

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
