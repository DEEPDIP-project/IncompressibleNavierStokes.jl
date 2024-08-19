using Adapt
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

plotdir = "output/postanalysis/plots"
outdir = "output/postanalysis"

seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    prior = 345, # A-priori training batch selection
    post = 456, # A-posteriori training batch selection
)

using LuxCUDA
using CUDA
T = Float32
ArrayType = CuArray
CUDA.allowscalar(true)
device = x -> adapt(CuArray, x)
clean() = (GC.gc(); CUDA.reclaim())

#cpu option
#T = Float32
#ArrayType = Array
#device = identity
#clean() = nothing

rng = Random.Xoshiro()
Random.seed!(rng, seeds.dns)

# Parameters
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

# Load IO arrays
io_train = load("$outdir/io_train.jld2", "io_train")
#io_valid = load("$outdir/io_valid.jld2", "io_valid")
#io_test = load("$outdir/io_test.jld2", "io_test")

# Load model params
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

# Priori: Load learned parameters and training times
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

# Smagorinsky model: Load trained parameters
smag = load("$outdir/smag.jld2")["smag"];
θ_smag = map(s -> s.θ, smag)

# ## Outputs (complete solution)
u_output = let
    ngrid, nfilter = size(io_train)
    temp = []
    t = data_test.t
    nupdate = 2
    Δt = (t[2] - t[1]) / nupdate
    temp = [ntuple(α -> zeros(T, 0, 0), 2)]
    u_ref = fill(temp, ngrid, nfilter)
    u_nomodel = fill(temp, ngrid, nfilter)
    u_smag = fill(temp, ngrid, nfilter, 2)
    u_cnn_prior = fill(temp, ngrid, nfilter, 2)
    u_cnn_post = fill(temp, ngrid, nfilter, 2)
    for iorder = 1:2, ifil = 1:nfilter, igrid = 1:ngrid
        clean()
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        setup = setups_test[igrid]
        psolver = psolver_spectral(setup)
        ustart = data_test.data[igrid, ifil].u[1] |> device
        tlims = (t[1], t[end])
        T = eltype(ustart[1])
        s(closure_model, θ) =
            solve_unsteady(;
                setup=(; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(RK44(; T), getorder(iorder)),
                Δt,
                psolver,
                θ,
                processors = (
                    field = IncompressibleNavierStokes.fieldsaver(; setup, nupdate = nupdate),
                    log = IncompressibleNavierStokes.timelogger(; nupdate = nupdate)
                )
            )[2].field # export only the outputs
        if iorder == 1
            u_ref[igrid, ifil] = data_test.data[igrid, ifil].u
            ## Does not depend on projection order
            u_nomodel[igrid, ifil] = [item.u for item in s(nothing, nothing)]
        end
        u_smag[igrid, ifil, iorder] = [item.u for item in
            s(IncompressibleNavierStokes.smagorinsky_closure(setup), θ_smag[ifil, iorder])]
        u_cnn_prior[igrid, ifil, iorder] = [item.u for item in
            s(wrappedclosure(closure, setup), θ_cnn_prior[igrid, ifil])]
        u_cnn_post[igrid, ifil, iorder] = [item.u for item in
            s(wrappedclosure(closure, setup), θ_cnn_post[igrid, ifil, iorder])]
    end
    (; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post)
end;
clean();

# Save solution
jldsave("$savepath/u_output.jld2"; u_output)
