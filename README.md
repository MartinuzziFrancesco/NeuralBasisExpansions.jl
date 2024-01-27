# NBeats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MartinuzziFrancesco.github.io/NBeats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MartinuzziFrancesco.github.io/NBeats.jl/dev/)
[![Build Status](https://github.com/MartinuzziFrancesco/NBeats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MartinuzziFrancesco/NBeats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MartinuzziFrancesco/NBeats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinuzziFrancesco/NBeats.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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
