using Adapt
using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes.RKMethods
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using NeuralClosure
using NNlib
using Optimisers
using PaperDC
using Random
using SparseArrays
using FFTW


# Color palette for consistent theme throughout paper
palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])

# Encode projection order ("close first, then project" etc)
getorder(i) =
    if i == 1
        :first
    elseif i == 2
        :last
    elseif i == 3
        :second
    else
        error("Unknown order: $i")
    end

# Choose where to put output
#plotdir = "output/postanalysis/plots"
#outdir = "output/postanalysis"
outdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo"
plotdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo/plots"

# ## Hardware selection
T = Float32
ArrayType = Array
device = identity
clean() = nothing

seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    prior = 345, # A-priori training batch selection
    post = 456, # A-posteriori training batch selection
)

# Parameters
rng = Random.Xoshiro()
Random.seed!(rng, seeds.dns)
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n))(4096), # DNS resolution
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    create_psolver = psolver_spectral,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng,
)

# Get parameters for multiple LES resolutions
params_train = (; get_params([64, 128, 256])..., tsim = T(0.5), savefreq = 10);
params_valid = (; get_params([64, 128, 256])..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 10);

# Load filtered DNS data
#data_train = load("$outdir/data_train.jld2", "data_train");
#data_valid = load("$outdir/data_valid.jld2", "data_valid");
data_test = load("$outdir/data_test.jld2", "data_test");

# Load IO arrays
io_train = load("$outdir/io_train.jld2", "io_train")
#io_valid = load("$outdir/io_valid.jld2", "io_valid")
#io_test = load("$outdir/io_test.jld2", "io_test")

# Build LES setup and assemble operators
getsetups(params) = [
    Setup(
        ntuple(α -> LinRange(T(0), T(1), nles[α] + 1), params.D)...;
        params.Re,
        params.ArrayType,
    ) for nles in params.nles
]
setups_train = getsetups(params_train);
setups_valid = getsetups(params_valid);
setups_test = getsetups(params_test);

# Neural Networks
rng = Random.Xoshiro(seeds.θ₀)
# CNN architecture 1
# mname = "balzac"
# closure, θ₀ = cnn(;
#     setup = setups_train[1],
#     radii = [2, 2, 2, 2],
#     channels = [20, 20, 20, params_train.D],
#     activations = [leakyrelu, leakyrelu, leakyrelu, identity],
#     use_bias = [true, true, true, false],
#     rng,
# );

# CNN architecture 2
mname = "rimbaud"
closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2, 2],
    channels = [24, 24, 24, 24, params_train.D],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng,
);
savepath = "$outdir/$mname"
closure.chain

# Priori: Load learned parameters and training times
priorfiles = map(CartesianIndices(io_train)) do I
    ig, ifil = I.I
    "$savepath/prior_ifilter$(ifil)_igrid$(ig).jld2"
end
prior = map(f -> load(f)["prior"], priorfiles)
θ_cnn_prior = [copyto!(device(θ₀), p.θ) for p in prior];

# Posteriori: Load learned parameters and training times
postfiles = map(CartesianIndices((size(io_train)..., 2))) do I
    ig, ifil, iorder = I.I
    "$savepath/post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2"
end
post = map(f -> load(f)["post"], postfiles);
θ_cnn_post = [copyto!(device(θ₀), p.θ) for p in post];
size(θ_cnn_post)

# Smagorinsky model: Load trained parameters
smag = load("$outdir/smag.jld2")["smag"];
θ_smag = map(s -> s.θ, smag)

# Get Solutions
setup = setups_test[1]
ustart = data_test.data[1, 1].u[1]
t = data_test.t
tlims = (t[1], t[end])
psolver = psolver_spectral(setup)
nupdate = 2
Δt = (t[2] - t[1]) / nupdate
s(closure_model, θ) = 
    solve_unsteady(; setup=(; setup..., closure_model),
        ustart,
        tlims,
        method = RKProject(RK44(; T), getorder(1)),
        Δt,
        psolver,
        θ,
        processors = (
            field = IncompressibleNavierStokes.fieldsaver(; setup, nupdate = nupdate),
            log = IncompressibleNavierStokes.timelogger(; nupdate = nupdate)
        )
    );

(state, outputs) = s(wrappedclosure(closure, setup), θ_cnn_prior[1,1])

wc = wrappedclosure(closure, setup)
state, outputs = solve_unsteady(; setup=(; setup..., wc),
        ustart,
        tlims,
        method = RKProject(RK44(; T), getorder(1)),
        Δt,
        psolver,
        θ = θ_cnn_post[1,2,1],
        processors = (
            field = IncompressibleNavierStokes.fieldsaver(; setup, nupdate = nupdate),
            log = IncompressibleNavierStokes.timelogger(; nupdate = nupdate)
        )
    );

# then one can access each time step as:
outputs.field[4].u[1]
outputs.field
u_vector = [item.u for item in outputs.field]
typeof(u_vector[2])
# last u:
state.u[1]

# Exploration of datasets - test
# data_test.data[igrid, ifil].u[step][u or p]
data_test.data[3, 1].u[2][1]
size(data_test.data) # has 5 grids, 2 bigger than the ones in train and validation datasets.

# what about different parameters? or a different initial condition from that of the test dataset.
# here we are testing a different grid (5th of test dataset: 1024) with the params trained for a grid of 256 and 2nd order (best results).
setup = setups_test[5]
wc = wrappedclosure(closure, setup)
state, outputs = solve_unsteady(; setup=(; setup..., wc),
    ustart=data_test.data[5, 1].u[1],
    tlims,
    method = RKProject(RK44(; T), getorder(2)),
    Δt,
    psolver=psolver_spectral(setup),
    θ = θ_cnn_post[3,1,2],
    processors = (
        field = IncompressibleNavierStokes.fieldsaver(; setup=setup, nupdate = nupdate),
        log = IncompressibleNavierStokes.timelogger(; nupdate = nupdate)
    )
    );

u_vector = [item.u for item in outputs.field]

function visualize(field, setup; save_path = nothing)
    GLMakie.activate!()
    u = field[1]
    o = Observable((; u, temp = nothing, t = nothing))
    # default field is vorticity but can be changed to velocitynorm or 1. 
    scene = fieldplot(o; setup,
        #fieldname = :velocitynorm,
        #fieldname = 1,
    )
    if save_path !== nothing
        record(scene, save_path, 1:length(field); framerate = 30) do i
            o[] = (; o[]..., u = field[i])
        end
    else 
        scene |> display
        for i = 1:length(field)
            o[] = (; o[]..., u = field[i])
            sleep(0.001)
        end
    end
end

field, setup = u_vector, setup;
visualize(field, setup)
visualize(field, setup; save_path="$plotdir/post_2_1024.mp4")
field = data_test.data[5, 1].u
visualize(field, setup; save_path="$plotdir/ref_1024.mp4")

# error
field = [(a .- b) for (a, b) in zip(u_vector, data_test.data[5, 1].u[2:end])]
visualize(field, setup; save_path="$plotdir/post_2_1024_error.mp4")