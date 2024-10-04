# Note: there is no julia project to run this script (i.e. in this folder).
# Suggestion: use PaperDC environment:
# julia
#]
# activate /Users/luisaorozco/Documents/Projects/DEEPDIP/fork_INS/lib/PaperDC
# # exit Pkg mode
# include("/Users/luisaorozco/Documents/Projects/DEEPDIP/fork_INS/lib/PaperDC/setup.jl")

# After this you should have dependencies suchs as IncompressibleNavierStokes, NeuralClosure and PaperDC available.

using GLMakie: GLMakie
using JLD2: load
device = identity
T = Float32
ArrayType = Array

mname = "rimbaud"
#mname = "balzac"
outdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo"
savepath = "$outdir/$mname"
plotdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo/plots"

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

# normal preparation of params and setups
using PaperDC
using NeuralClosure
using IncompressibleNavierStokes
using IncompressibleNavierStokes.RKMethods
using Random: Random
rng = Random.Xoshiro()
seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    prior = 345, # A-priori training batch selection
    post = 456, # A-posteriori training batch selection
)
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
params_train = (; get_params([64, 128, 256])..., tsim = T(0.5), savefreq = 10);
params_valid = (; get_params([64, 128, 256])..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 10);

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

io_train = load("$outdir/data/io_train.jld2", "io_train")
io_test = load("$outdir/data/io_test.jld2", "io_test")

if mname == "balzac"
    closure, θ₀ = cnn(;
        setup = setups_train[1],
        radii = [2, 2, 2, 2],
        channels = [20, 20, 20, params_train.D],
        activations = [leakyrelu, leakyrelu, leakyrelu, identity],
        use_bias = [true, true, true, false],
        rng,
    );
else
    closure, θ₀ = cnn(;
        setup = setups_train[1],
        radii = [2, 2, 2, 2, 2],
        channels = [24, 24, 24, 24, params_train.D],
        activations = [tanh, tanh, tanh, tanh, identity],
        use_bias = [true, true, true, true, false],
        rng,
    );
end
closure.chain

# Load learned parameters and training times
priorfiles = map(CartesianIndices(io_train)) do I
    ig, ifil = I.I
    "$savepath/prior_ifilter$(ifil)_igrid$(ig).jld2"
end
prior = map(f -> load(f)["prior"], priorfiles)
θ_cnn_prior = [copyto!(device(θ₀), p.θ) for p in prior];

postfiles = map(CartesianIndices((size(io_train)..., 2))) do I
    ig, ifil, iorder = I.I
    "$savepath/post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2"
end
post = map(f -> load(f)["post"], postfiles);
θ_cnn_post = [copyto!(device(θ₀), p.θ) for p in post];

smag = load("$outdir/smag.jld2")["smag"];
θ_smag = map(s -> s.θ, smag);
