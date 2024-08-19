# this test is for the NODE functionality of Syver repo
using Makie
using IncompressibleNavierStokes

T = Float32
ArrayType = Array
Re = T(1_000)
n = 64
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x...; Re, ArrayType);
ustart = random_field(setup, T(0));
dt = T(1e-3)
trange = (T(0), T(1e-3))
trange = (T(0), T(1))
savevery = 20
saveat = 20*dt

#@time state, outputs = solve_unsteady(;
#    setup,
#    ustart,
#    tlims = trange,
#    Δt = dt,
#    processors = (
#        ehist = realtimeplotter(;
#            setup,
#            plot = energy_history_plot,
#            nupdate = 10,
#            displayfig = false,
#        ),
#        anim = animator(; setup, path = "./vorticity.mkv", nupdate = savevery),
#        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
#        log = timelogger(; nupdate = 100),
#    ),
#);
using Lux
using Random
using CUDA
using KernelAbstractions
using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum!, apply_bc_u!, project!
# if you run setup.jl you need only to run this line
using NeuralClosure
# otherwise you can also include the memeber manually
include("src/closure.jl")
include("src/cnn.jl")
include("src/create_les_data.jl")
include("src/filter.jl")
#include("src/fno.jl")
include("src/groupconv.jl")
include("src/NeuralClosure.jl")
include("src/training.jl")

rng = Random.default_rng()
les_data = create_les_data(
    D = 2,
    Re = Re,
    rng = rng,
)
keys(les_data)
size(les_data.data)
les_data.data[1]
keys(les_data.data[1])
les_data

#filtered u: Φu
size(les_data.data[1][:u]) #1000 time steps
typeof(les_data.data[1][:u][1]) #each time step is a tuple of 2 Matrices
size(les_data.data[1][:u][1][1]) # each matrix is 66x66, nles is by default 64 and the 2 extra must be "ghost" point for boundary conditions
size(les_data.data[1][:u][1][2])

#closure (labels) have the same structure as u
size(les_data.data[1][:c])

# Dataset
io_prior = create_io_arrays([les_data], [setup])
size(io_prior)

#1st attempt
#dataloader_prior = create_dataloader_prior(les_data.data[1]; rng=rng)
#x_prior, y_prior = dataloader_prior()
#length(x_prior) #batch size is 50
#x_prior[1] # each sample is a tuple of 2 matrices, I guess x would be :u and y would be :c

dataloader_prior = create_dataloader_prior(io_prior[1]; rng)
x_prior, y_prior = dataloader_prior()
size(x_prior) #(nles, nles, 2, batch_size)

# Neural network (CNN)
model_prior, θ_prior = cnn(;
    setup=setup,
    radii=[2, 2, 2, 2, 2],
    channels=[2, 8, 8, 8, 2], #2 dimensions `D`
    activations=[leakyrelu, leakyrelu, leakyrelu, leakyrelu, identity] ,
    use_bias=[true, true, true, true, false],
    channel_augmenter = identity,
    rng = rng)

model_prior.chain #only to visualize the model nicer?

# Loss
loss_prior = create_loss_prior(mean_squared_error, model_prior)

using Optimisers
opt_prior = Optimisers.setup(Adam(T(1.0f-4)), θ_prior);

#training
# have to solve this by for example: using GLMakie, for the moment skip
#relerr_prior = create_relerr_prior(model_prior, x_prior, y_prior) # wrong: has to be vs validation set
#callback = create_callback(relerr_prior; θ=θ_prior, display_each_iteration = true)

(; optstate, θ, callbackstate) = train(
        [dataloader_prior],
        loss_prior,
        opt_prior,
        θ_prior;
        niter = 100
    );
#use best parameters instead of last (θ)
θ_prior = callbackstate.θmin # only works if there is a callback saving this
