using JLD2
using GLMakie
import Random
using NeuralClosure

# Choose where to put output
#plotdir = "output/postanalysis/plots"
#outdir = "output/postanalysis"
outdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo"
plotdir = "/Users/luisaorozco/Documents/Projects/DEEPDIP/demo/plots"

# Select model
#mname = "balzac"
mname = "rimbaud"

# Hardware selection
T = Float32
ArrayType = Array
device = identity
clean() = nothing

# Parameters and setup
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
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 10);

getsetups(params) = [
    Setup(
        ntuple(α -> LinRange(T(0), T(1), nles[α] + 1), params.D)...;
        params.Re,
        params.ArrayType,
    ) for nles in params.nles
]
setups_test = getsetups(params_test);

# Load solutions: either just the final or the complete history
#(; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post) = load("$outdir/$mname/ufinal.jld2", "ufinal");
(; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post) = load("$outdir/$mname/u_output.jld2", "u_output");

# u_ref and u_nomodel have the format [n_grids, n_filters], u_ref has an extra time-element: initial condition.
# u_cnn_prior, u_cnn_post and u_smag have the format [n_grids, n_filters, n_order] although the order is only > 1 for posteriori.

# Visualizations
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

# Bare fields
field, setup = u_ref[3, 1], setups_test[3];
#visualize(field, setup)
#visualize(field, setup; save_path="$plotdir/ref.gif")
#field = u_nomodel[3, 1]
#visualize(field, setup; save_path="$plotdir/no_model.gif")
field = u_cnn_prior[3, 1, 1]
visualize(field, setup; save_path="$plotdir/prior.gif")
field = u_cnn_post[3, 1, 1]
visualize(field, setup; save_path="$plotdir/post_1.gif")
field = u_cnn_post[3, 1, 2]
visualize(field, setup; save_path="$plotdir/post_2.gif")

# Error fields
field = [(a .- b) for (a, b) in zip(u_nomodel[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/nomodel_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_prior[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/prior_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_post[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/post_1_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_post[3, 1, 2], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/post_2_error.mp4")