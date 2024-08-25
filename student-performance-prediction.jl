import Pkg
using Pkg
Pkg.add("PyCall")
ENV["PYTHON"] = "C:\\Users\\seyma\\Anaconda3\\python.exe"  # or the path to your Python executable
Pkg.build("PyCall")
using PyCall
py"print('Hello from Python!')"

using Pkg
Pkg.add("LibGit2")

import Pkg
using Flux
using Zygote
using MLDatasets
using MLUtils
using Plots
using NNlib  # activation functions
using StatsBase
using DataFrames
using Random
using CSV
using PyCall
Pkg.add("Conda")
using Conda
Conda.add("scikit-learn")


#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 24016569856 ## <---replace 0 by your student_number 
### ---- ###
## In this HW you are on your own, 

# Load the Student Performance dataset from my Desktop
data_path = "C:\\Users\\seyma\\OneDrive\\Masaüstü\\Student_Performance.csv"  # Update this path to your actual file
df = CSV.read(data_path, DataFrame)

# Drop the "Extracurricular Activities" column because it is not so much meaningful to my regression problem and
# it has a type "String3". So, it may cause a problem to me.
select!(df, Not("Extracurricular Activities"))


# Preprocess the data
data_x = Matrix(df[:, 1:end-1]) |> Matrix{Float32} |> transpose
data_y = Array(df[:, end]) |> Array{Float32}

data_x = (data_x .- mean(data_x, dims = 2)) ./ std(data_x, dims = 2)
data_y = (data_y .- mean(data_y)) ./ std(data_y)

# Firstly, let's add 2 hidden layer and set up learning rates as 0.05 and 0.005. (Here, I have two different setup.)
function train_network(hidden_layers, learning_rate, epochs)
    Network = Chain(
        Dense(size(data_x, 1), hidden_layers[1], relu),
        Dropout(0.5),
        Dense(hidden_layers[1], hidden_layers[2], relu),
        Dropout(0.5),
        Dense(hidden_layers[2], 1)
    )
    
    opt_state = Flux.setup(Flux.Optimise.Momentum(learning_rate), Network)

    indexes = collect(1:size(data_x, 2))
    Random.seed!(0)
    shuffle!(indexes)

    train_indexes, val_indexes = indexes[round(Int, 0.2*length(indexes)):end], indexes[1:round(Int, 0.2*length(indexes))-1]
    X_train, X_val, y_train, y_val = data_x[:, train_indexes], data_x[:, val_indexes], data_y[train_indexes], data_y[val_indexes]

    train_data = DataLoader((X_train, y_train), batchsize = 16)
    val_data = DataLoader((X_val, y_val), batchsize = 8)

    train_losses = []
    val_losses = []

    for i in 1:epochs
        temp_val = 0.0f0
        temp_size = 0
        trainmode!(Network)
        for (x, y) in train_data
            val, grads = Zygote.withgradient(Network) do Network
                mean(abs.(Network(x) .- transpose(y)))
            end 
            temp_val += val
            temp_size += size(x)[end]
            Flux.update!(opt_state, Network, grads[1])
        end
        temp_validation_val = 0.0f0
        temp_validation_size = 0
        testmode!(Network)
        for (x, y) in val_data
            temp_validation_val += mean((Network(x) .- transpose(y)).^2)
            temp_validation_size += size(x)[end]
        end
        push!(train_losses, temp_val/temp_size)
        push!(val_losses, temp_validation_val/temp_validation_size)
        println("Epoch $i: train loss = $(temp_val/temp_size), validation loss = $(temp_validation_val/temp_validation_size)")
    end

    skmetrics = pyimport("sklearn.metrics")
    r2 = skmetrics.r2_score(y_val, Network(X_val) |> transpose)
    println("R2 score: $r2")
    return Network, r2, train_losses, val_losses
end


# Experiment the setups. 
setup1 = ([13, 10], 0.01)
setup2 = ([13, 20], 0.005)

epochs = 1000
println("Setup 1:")
network1, r2_1, train_losses1, val_losses1 = train_network(setup1[1], setup1[2], epochs)
# Epoch 1000: train loss = 0.02495917, validation loss = 0.015032917
# R2 score: 0.8780410724752464

println("\nSetup 2:")
network2, r2_2, train_losses2, val_losses2 = train_network(setup2[1], setup2[2], epochs)
# Epoch 1000: train loss = 0.019111458, validation loss = 0.00930998
# R2 score: 0.9244793374960797


# Now, change the shape of the network. Remove one more hidden layer
# (because my data is not soo complex and feature number is few.)
function train_network_1(hidden_layers, learning_rate, epochs)
    Network = Chain(
        Dense(size(data_x, 1), hidden_layers[1], relu),
        Dropout(0.5),
        Dense(hidden_layers[1], 1)
    )
    
    opt_state = Flux.setup(Flux.Optimise.Momentum(learning_rate), Network)

    indexes = collect(1:size(data_x, 2))
    Random.seed!(0)
    shuffle!(indexes)

    train_indexes, val_indexes = indexes[round(Int, 0.2*length(indexes)):end], indexes[1:round(Int, 0.2*length(indexes))-1]
    X_train, X_val, y_train, y_val = data_x[:, train_indexes], data_x[:, val_indexes], data_y[train_indexes], data_y[val_indexes]

    train_data = DataLoader((X_train, y_train), batchsize = 16)
    val_data = DataLoader((X_val, y_val), batchsize = 8)

    train_losses = []
    val_losses = []

    for i in 1:epochs
        temp_val = 0.0f0
        temp_size = 0
        trainmode!(Network)
        for (x, y) in train_data
            val, grads = Zygote.withgradient(Network) do Network
                mean(abs.(Network(x) .- transpose(y)))
            end 
            temp_val += val
            temp_size += size(x)[end]
            Flux.update!(opt_state, Network, grads[1])
        end
        temp_validation_val = 0.0f0
        temp_validation_size = 0
        testmode!(Network)
        for (x, y) in val_data
            temp_validation_val += mean((Network(x) .- transpose(y)).^2)
            temp_validation_size += size(x)[end]
        end
        push!(train_losses, temp_val/temp_size)
        push!(val_losses, temp_validation_val/temp_validation_size)
        println("Epoch $i: train loss = $(temp_val/temp_size), validation loss = $(temp_validation_val/temp_validation_size)")
    end

    skmetrics = pyimport("sklearn.metrics")
    r2 = skmetrics.r2_score(y_val, Network(X_val) |> transpose)
    println("R2 score: $r2")
    return Network, r2, train_losses, val_losses
end


# Experiment that setup with learning rate: 0.005. (Because, in the above scenerio l.r=0.005 gives more accurate result.)  
setup3 = ([13, 15], 0.005)

println("\nSetup 3:")
network3, r2_3, train_losses3, val_losses3 = train_network_1(setup3[1], setup3[2], epochs)
# Epoch 1000: train loss = 0.017619107, validation loss = 0.0037267257
# R2 score: 0.9697640208866516


println("R2 Scores for different setups:")
println("Setup 1: $r2_1")
println("Setup 2: $r2_2")
println("Setup 3: $r2_3")


using Plots

# Plotting function
function plot_losses(train_losses, val_losses; title="")
    plot(
        collect(1:length(train_losses)), train_losses, label="Training Loss",
        xlabel="Epoch", ylabel="Loss", title=title, lw=2,
    )
    plot!(
        collect(1:length(val_losses)), val_losses, label="Validation Loss", lw=2,
    )
end

# Plot for Setup 1
plot_losses(train_losses1, val_losses1, title="Setup 1 Losses")
savefig("C:\\Users\\seyma\\OneDrive\\Masaüstü\\setup1_losses.pdf") 

# Plot for Setup 2
plot_losses(train_losses2, val_losses2, title="Setup 2 Losses")
savefig("C:\\Users\\seyma\\OneDrive\\Masaüstü\\setup2_losses.pdf") 
# Plot for Setup 3
plot_losses(train_losses3, val_losses3, title="Setup 3 Losses")
savefig("C:\\Users\\seyma\\OneDrive\\Masaüstü\\setup3_losses.pdf") 


# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
    @assert student_number != 0
    println("Seems everything is ok!!!")
end
