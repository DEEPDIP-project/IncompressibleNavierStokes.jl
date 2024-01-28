# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Train closure model
#
# Here, we consider a periodic box ``[0, 1]^2``. It is discretized with a
# uniform Cartesian grid with square cells.

using Adapt
using CairoMakie
using GLMakie
using IncompressibleNavierStokes
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote
using SparseArrays
using KernelAbstractions
using FFTW

GLMakie.activate!()

set_theme!(; GLMakie = (; scalefactor = 1.5))

output = "../SupervisedClosure/figures/"

# Random number generator
rng = Random.default_rng()
Random.seed!(rng, 12345)

# Floating point precision
T = Float64

# Array type
ArrayType = Array
device = identity
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
# T = Float64;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = x -> adapt(CuArray, x)

# Parameters
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar),
    # ndns = (n -> (n, n))(1024),
    # ndns = (n -> (n, n))(2048),
    ndns = (n -> (n, n))(4096),
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    PSolver = SpectralPressureSolver,
    icfunc = (setup, psolver) -> random_field(
        setup,
        zero(eltype(setup.grid.x[1]));
        # A = 1,
        kp = 20,
        psolver,
    ),
)

params_train = (; get_params([64, 128, 256])..., savefreq = 10);
params_valid = (; get_params([64, 128, 256])..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.2), savefreq = 20);

# Create LES data from DNS
data_train = [create_les_data(; params_train...) for _ = 1:5];
data_valid = [create_les_data(; params_valid...) for _ = 1:1];
data_test = create_les_data(; params_test...);

# Save filtered DNS data
jldsave("output/divfree/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/divfree/data.jld2", "data_train", "data_valid", "data_test");

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

data_train[1].t
data_train[1].data |> size
data_train[1].data[1, 1].u[end][1]

# Create input/output arrays
io_train = create_io_arrays(data_train, setups_train);
io_valid = create_io_arrays(data_valid, setups_valid);

# jldsave("output/divfree/io_train.jld2"; io_train)
# jldsave("output/divfree/io_train.jld2"; io_valid)

io_train[1].u |> extrema
io_train[1].c |> extrema
io_valid[1].u |> extrema
io_valid[1].c |> extrema

# Inspect data
ig = 2
ifil = 1
field, setup = data_train[1].data[ig, ifil].u, setups_train[ig];
# field, setup = data_valid[1].data[ig, ifil].u, setups_valid[ig];
# field, setup = data_test.data[ig, ifil], setups_test[ig];
u = device.(field[1]);
o = Observable((; u, t = nothing));
# energy_spectrum_plot(o; setup)
fieldplot(
    o;
    setup,
    # fieldname = :velocity,
    # fieldname = 1,
)
for i = 1:length(field)
    o[] = (; o[]..., u = device(field[i]))
    sleep(0.001)
end

GLMakie.activate!()
CairoMakie.activate!()

# Training data plot
ifil = 1
boxx = T(0.3), T(0.5)
boxy = T(0.5), T(0.7)
box = [
    Point2f(boxx[1], boxy[1]),
    Point2f(boxx[2], boxy[1]),
    Point2f(boxx[2], boxy[2]),
    Point2f(boxx[1], boxy[2]),
    Point2f(boxx[1], boxy[1]),
]
# fig = with_theme() do
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    sample = data_train[1]
    fig = Figure()
    for (i, it) in enumerate((1, length(sample.t)))
        # for (j, ig) in enumerate((1, 2, 3))
        for (j, ig) in enumerate((1, 2))
            setup = setups_train[ig]
            xf = Array.(getindex.(setup.grid.xp, setup.grid.Ip.indices))
            u = sample.data[ig, ifil].u[it] |> device
            ωp =
                IncompressibleNavierStokes.interpolate_ω_p(
                    IncompressibleNavierStokes.vorticity(u, setup),
                    setup,
                )[setup.grid.Ip] |> Array
            colorrange = IncompressibleNavierStokes.get_lims(ωp)
            opts = (;
                xticksvisible = false,
                xticklabelsvisible = false,
                yticklabelsvisible = false,
                yticksvisible = false,
            )
            i == 2 && (
                opts = (;
                    opts...,
                    xlabel = "x",
                    xticksvisible = true,
                    xticklabelsvisible = true,
                )
            )
            j == 1 && (
                opts = (;
                    opts...,
                    ylabel = "y",
                    yticklabelsvisible = true,
                    yticksvisible = true,
                )
            )
            ax = Axis(
                fig[i, j];
                opts...,
                title = "n = $(params_train.nles[ig]), t = $(round(sample.t[it]; digits = 1))",
                aspect = DataAspect(),
                limits = (T(0), T(1), T(0), T(1)),
            )
            heatmap!(ax, xf..., ωp; colorrange)
            # lines!(ax, box; color = Cycled(2))
        end
    end
    fig
end

save("$output/training_data.pdf", fig)

closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2],
    channels = [20, 20, 20, params_train.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
);
closure.chain

# Train grid-specialized closure models
θ_cnn = map(CartesianIndices(size(io_train))) do I
    # Prepare training
    ig, ifil = I.I
    @show ig ifil
    d = create_dataloader_prior(io_train[ig, ifil]; batchsize = 50, device)
    θ = T(1.0e0) * device(θ₀)
    loss = createloss(mean_squared_error, closure);
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    callbackstate = Point2f[]
    it = rand(1:size(io_valid[ig, ifil].u, 4), 50)
    validset = map(v -> v[:, :, :, it], io_valid[ig, ifil])
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 10000,
        ncallback = 20,
        callbackstate,
        callback = create_relerr_prior(closure, validset),
    )
    θ
end
GC.gc()
CUDA.reclaim()

# # Save trained parameters
# jldsave("output/divfree/theta_prior.jld2"; theta = Array.(θ_cnn))

# # Load trained parameters
# θ_cnn = [device(θ₀) for _ in CartesianIndices(size(data_train[1].data))];
# θθ = load("output/divfree/theta_prior.jld2");
# copyto!.(θ_cnn, θθ["theta"]);

# θ_post = map(CartesianIndices(size(io_train))) do I
#     ig, ifil = I.I
θ_post = let ig = 2, ifil = 1
    iorder = 1
    setup = setups_train[ig]
    psolver = SpectralPressureSolver(setup)
    loss = IncompressibleNavierStokes.create_loss_post(;
        setup,
        psolver,
        closure,
        nupdate = 4,
        unproject_closure = iorder == 2,
    )
    data = [(; u = d.data[ig, ifil].u, d.t) for d in data_train]
    d = create_dataloader_post(data; device, nunroll = 10)
    # θ = copy(θ_cnn[ig, ifil])
    θ = device(θ₀)
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    callbackstate = Point2f[]
    it = rand(1:size(io_valid[ig, ifil].u, 4), 50)
    data = data_valid[1]
    data = (; u = device.(data.data[ig, ifil].u), data.t)
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 100,
        ncallback = 1,
        callbackstate,
        callback = create_callback(
            create_relerr_post(;
                data,
                setup,
                psolver,
                closure,
                unproject_closure = iorder == 2,
            );
            state = callbackstate,
            displayref = false,
        ),
    )
    θ
end

# Train Smagorinsky closure model
ig = 2;
setup = setups_train[ig];
sample = data_train[1];
m = smagorinsky_closure(setup);
θ = T(0.05)
e_smag = sum(2:length(sample.t)) do it
    It = setup.grid.Ip
    u = sample.u[ig][it] |> device
    c = sample.c[ig][it] |> device
    mu = m(u, θ)
    e = zero(eltype(u[1]))
    for α = 1:D
        # e += sum(abs2, mu[α][Ip] .- c[α][Ip]) / sum(abs2, c[α][Ip])
        e += norm(mu[α][Ip] .- c[α][Ip]) / norm(c[α][Ip])
    end
    e / D
end / length(sample.t)
# for θ in LinRange(T(0), T(1), 100)];

# lines(LinRange(T(0), T(1), 100), e_smag)

# Errors for grid-specialized closure models
e_smag, e_cnn = let
    e_smag = zeros(T, size(data_test.data)..., 2)
    e_cnn = zeros(T, size(data_test.data)..., 2)
    for iorder = 1:2, ifil = 1:2, ig = 1:size(data_test.data, 1)
        @show iorder ifil ig
        setup = setups_test[ig]
        params = params_test[ig]
        psolver = SpectralPressureSolver(setup)
        u = device.(data_test.data[ig, ifil].u)
        u₀ = device(data_test.data[ig, ifil].u[1])
        Δt = params_test.Δt * params_test.savefreq
        tlims = extrema(data_test.t)
        nupdate = 20
        Δt /= nupdate
        processors = (; relerr = relerr_trajectory(u, setup; nupdate))
        # Smagorinsky
        closedsetup = (;
            setup...,
            closure_model = smagorinsky_closure(setup),
            unproject_closure = iorder == 2,
        )
        _, outputs =
            solve_unsteady(closedsetup, u₀, tlims; Δt, psolver, processors, θ = T(0.1))
        e_smag[ig, ifil, iorder] = outputs.relerr[]
        # CNN
        # The first grids are trained for
        if ig ≤ size(data_train[1].data, 1)
            closedsetup = (;
                setup...,
                closure_model = wrappedclosure(closure, setup),
                unproject_closure = iorder == 2,
            )
            _, outputs = solve_unsteady(
                closedsetup,
                u₀,
                tlims;
                Δt,
                psolver,
                processors,
                θ = θ_cnn[ig, ifil],
            )
            e_cnn[ig, ifil, iorder] = outputs.relerr[]
        end
    end
    e_smag, e_cnn
end
e_smag
e_cnn

# julia> e_smag                julia> e_smag
# 5×2×2 Array{Float32, 3}:     5×2×2 Array{Float32, 3}:
# [:, :, 1] =                  [:, :, 1] =
#  0.781462   0.778116          0.781462   0.778104
#  0.505524   0.505589          0.505524   0.505565
#  0.30132    0.301625          0.30132    0.301601
#  0.143863   0.144141          0.143863   0.144118
#  0.0457445  0.0459661         0.0457429  0.0459541
#                                                       
# [:, :, 2] =                  [:, :, 2] =
#  0.813456   0.826474          0.813457   0.826474
#  0.527202   0.534727          0.527203   0.534726
#  0.318339   0.32539           0.318339   0.325389
#  0.152291   0.157393          0.152292   0.157392
#  0.0482445  0.051005          0.0482446  0.0510034
#                                                       
# julia> e_cnn                 julia> e_cnn
# 5×2×2 Array{Float32, 3}:     5×2×2 Array{Float32, 3}:
# [:, :, 1] =                  [:, :, 1] =
#  0.585886   0.5875            0.585887   0.5875
#  0.311889   0.277596          0.311892   0.277624
#  0.0725899  0.114744          0.0725901  0.114847
#  0.0        0.0               0.0        0.0
#  0.0        0.0               0.0        0.0
#                                                       
# [:, :, 2] =                  [:, :, 2] =
#    0.783691    0.859665         0.783688    0.859664
#    1.81263   NaN                1.78943     6.52473
#  NaN         NaN              NaN         NaN
#    0.0         0.0              0.0         0.0
#    0.0         0.0              0.0         0.0

# No model
e_nm = let
    e_nm = zeros(T, size(data_test.data)...)
    for ifil = 1:2, ig = 1:size(data_test.data, 1)
        @show ifil ig
        setup = setups_test[ig]
        params = params_test[ig]
        psolver = SpectralPressureSolver(setup)
        u = device.(data_test.data[ig, ifil].u)
        u₀ = device(data_test.data[ig, ifil].u[1])
        Δt = params_test.Δt * params_test.savefreq
        tlims = extrema(data_test.t)
        nupdate = 20
        Δt /= nupdate
        processors = (; relerr = relerr_trajectory(u, setup; nupdate))
        _, outputs = solve_unsteady(
            (; setup..., unproject_closure = false),
            u₀,
            tlims;
            Δt,
            psolver,
            processors,
        )
        e_nm[ig, ifil] = outputs.relerr[]
    end
    e_nm
end
e_nm

GC.gc()
CUDA.reclaim()

e_nm
e_smag
e_cnn

CairoMakie.activate!()
GLMakie.activate!()

# Plot convergence
with_theme(;
    # linewidth = 5,
    # markersize = 10,
    # markersize = 20,
    # fontsize = 20,
    palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"]),
) do
    iorder = 2
    lesmodel = iorder == 1 ? "closure-then-project" : "project-then-closure"
    nles = [n[1] for n in params_test.nles]
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        # xlabel = "n",
        # xlabel = L"\bar{n}",
        # title = "Relative error (DNS: n = $(params_test.ndns[1]))",
        title = "Relative error ($lesmodel)",
    )
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "No closure"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        scatterlines!(
            nles,
            e_nm[:, ifil];
            color = Cycled(1),
            linestyle,
            marker = :circle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "Smagorinsky"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        scatterlines!(
            nles,
            e_smag[:, ifil, iorder];
            color = Cycled(2),
            linestyle,
            marker = :utriangle,
            label,
        )
    end
    for ifil = 1:2
        ntrain = size(data_train[1].data, 1)
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        # ifil == 2 && (label = nothing)
        scatterlines!(
            nles[1:ntrain],
            e_cnn[1:ntrain, ifil, iorder];
            color = Cycled(3),
            linestyle,
            marker = :rect,
            label,
        )
    end
    # lines!(
    #     collect(extrema(nles[4:end])),
    #     n -> 2e4 * n^-2.0;
    #     linestyle = :dash,
    #     label = "n⁻²",
    #     color = Cycled(1),
    # )
    axislegend(; position = :rt)
    # iorder == 2 && limits!(ax, (T(60), T(1050)), (T(2e-2), T(1e1)))
    fig
end

e_cnn
e_cnn_2 = copy(e_cnn);
e_cnn_2[isnan.(e_cnn_2)] .= 1e10;
e_cnn_2;

save("$output/convergence_Lprior_prosecond.pdf", current_figure())
save("$output/convergence_Lprior_profirst.pdf", current_figure())

markers_labels = [
    (:circle, ":circle"),
    (:rect, ":rect"),
    (:diamond, ":diamond"),
    (:hexagon, ":hexagon"),
    (:cross, ":cross"),
    (:xcross, ":xcross"),
    (:utriangle, ":utriangle"),
    (:dtriangle, ":dtriangle"),
    (:ltriangle, ":ltriangle"),
    (:rtriangle, ":rtriangle"),
    (:pentagon, ":pentagon"),
    (:star4, ":star4"),
    (:star5, ":star5"),
    (:star6, ":star6"),
    (:star8, ":star8"),
    (:vline, ":vline"),
    (:hline, ":hline"),
    ('a', "'a'"),
    ('B', "'B'"),
    ('↑', "'\\uparrow'"),
    ('😄', "'\\:smile:'"),
    ('✈', "'\\:airplane:'"),
]

# Final spectra
ig = 4
setup = setups_test[ig];
params = params_test
pressure_solver = SpectralPressureSolver(setup);
uref = device(data_test.u[ig][end]);
u₀ = device(data_test.u[ig][1]);
Δt = params_test.Δt * params_test.savefreq;
tlims = extrema(data_test.t);
nupdate = 4;
Δt /= nupdate;
state_nm, outputs = solve_unsteady(setup, u₀, tlims; Δt, pressure_solver);
m = smagorinsky_closure(setup);
closedsetup = (; setup..., closure_model = u -> m(u, T(0.1)));
state_smag, outputs = solve_unsteady(closedsetup, u₀, tlims; Δt, pressure_solver);
# closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn_shared, setup));
closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn[ig-1], setup));
state_cnn, outputs = solve_unsteady(closedsetup, u₀, tlims; Δt, pressure_solver);

# Plot predicted spectra
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    (; xp, Ip) = setup.grid
    D = params.D
    K = size(Ip) .÷ 2
    kx = ntuple(α -> 0:K[α]-1, D)
    k = fill!(similar(xp[1], length.(kx)), 0)
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)
    # Sum or average wavenumbers between k and k+1
    nk = ceil(Int, maximum(k))
    kmax = nk - 1
    # kmax = minimum(K) - 1
    kint = 1:kmax
    ia = similar(xp[1], Int, 0)
    ib = sortperm(k)
    vals = similar(xp[1], 0)
    ksort = k[ib]
    jprev = 2 # Do not include constant mode
    for ki = 1:kmax
        j = findfirst(>(ki + 1), ksort)
        isnothing(j) && (j = length(k) + 1)
        ia = [ia; fill!(similar(ia, j - jprev), ki)]
        # val = doaverage ? T(1) / (j - jprev) : T(1)
        val = T(π) * ((ki + 1)^2 - ki^2) / (j - jprev)
        # val = T(1) / (j - jprev)
        vals = [vals; fill!(similar(vals, j - jprev), val)]
        jprev = j
    end
    ib = ib[2:jprev-1]
    A = sparse(ia, ib, vals, kmax, length(k))
    # Build inertial slope above energy
    # krange = [cbrt(T(kmax)), T(kmax)]
    krange = [T(kmax)^(T(2) / 3), T(kmax)]
    # slope, slopelabel = D == 2 ? (-T(3), L"k^{-3}") : (-T(5 / 3), L"k^{-5/3}")
    slope, slopelabel = D == 2 ? (-T(3), "|k|⁻³") : (-T(5 / 3), "|k|⁻⁵³")
    slopeconst = T(0)
    # Nice ticks
    logmax = round(Int, log2(kmax + 1))
    xticks = T(2) .^ (0:logmax)
    # Make plot
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xticks,
        xlabel = "|k|",
        ylabel = "e(|k|)",
        # title = "Kinetic energy (n = $(params.nles[ig])) at time t = $(round(data_test.t[end]; digits = 1))",
        title = "Kinetic energy (n = $(params.nles[ig]))",
        xscale = log10,
        yscale = log10,
        limits = (extrema(kint)..., T(1e-8), T(1)),
    )
    for (u, label) in (
        # (uref, "Reference"),
        (state_nm.u, "No closure"),
        (state_smag.u, "Smagorinsky"),
        (state_cnn.u, "CNN (specialized)"),
        (uref, "Reference"),
    )
        ke = IncompressibleNavierStokes.kinetic_energy(u, setup)
        e = ke[Ip]
        e = fft(e)[ntuple(α -> kx[α] .+ 1, D)...]
        e = abs.(e) ./ size(e, 1)
        e = A * reshape(e, :)
        ehat = max.(e, eps(T)) # Avoid log(0)
        slopeconst = max(slopeconst, maximum(ehat ./ kint .^ slope))
        lines!(ax, kint, Array(ehat); label)
    end
    inertia = 2 .* slopeconst .* krange .^ slope
    lines!(ax, krange, inertia; linestyle = :dash, label = slopelabel)
    axislegend(ax; position = :lb)
    autolimits!(ax)
    fig
end

save("predicted_spectra.pdf", fig)

# Plot spectrum errors
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    (; xp, Ip) = setup.grid
    D = params.D
    K = size(Ip) .÷ 2
    kx = ntuple(α -> 0:K[α]-1, D)
    k = fill!(similar(xp[1], length.(kx)), 0)
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)
    # Sum or average wavenumbers between k and k+1
    nk = ceil(Int, maximum(k))
    # kmax = nk - 1
    kmax = minimum(K) - 1
    kint = 1:kmax
    ia = similar(xp[1], Int, 0)
    ib = sortperm(k)
    vals = similar(xp[1], 0)
    ksort = k[ib]
    jprev = 2 # Do not include constant mode
    for ki = 1:kmax
        j = findfirst(>(ki + 1), ksort)
        isnothing(j) && (j = length(k) + 1)
        ia = [ia; fill!(similar(ia, j - jprev), ki)]
        # val = doaverage ? T(1) / (j - jprev) : T(1)
        val = T(π) * ((ki + 1)^2 - ki^2) / (j - jprev)
        vals = [vals; fill!(similar(vals, j - jprev), val)]
        jprev = j
    end
    ib = ib[2:jprev-1]
    A = sparse(ia, ib, vals, kmax, length(k))
    # Build inertial slope above energy
    # krange = [cbrt(T(kmax)), T(kmax)]
    krange = [T(kmax)^(T(2) / 3), T(kmax)]
    # slope, slopelabel = D == 2 ? (-T(3), L"k^{-3}") : (-T(5 / 3), L"k^{-5/3}")
    slope, slopelabel = D == 2 ? (-T(3), "|k|⁻³") : (-T(5 / 3), "|k|⁻⁵³")
    slopeconst = T(0)
    # Nice ticks
    logmax = round(Int, log2(kmax + 1))
    xticks = T(2) .^ (0:logmax)
    # Make plot
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xticks,
        xlabel = "|k|",
        ylabel = "e(|k|)",
        # title = "Kinetic energy (n = $(params.nles[ig])) at time t = $(round(data_test.t[end]; digits = 1))",
        title = "Relative energy error (n = $(params.nles[ig]))",
        xscale = log10,
        yscale = log10,
        limits = (extrema(kint)..., T(1e-8), T(1)),
    )
    ke = IncompressibleNavierStokes.kinetic_energy(uref, setup)
    e = ke[Ip]
    e = fft(e)[ntuple(α -> kx[α] .+ 1, D)...]
    e = abs.(e) ./ size(e, 1)
    e = A * reshape(e, :)
    eref = max.(e, eps(T)) # Avoid log(0)
    for (u, label) in (
        (state_nm.u, "No closure"),
        (state_smag.u, "Smagorinsky"),
        (state_cnn.u, "CNN (specialized)"),
    )
        ke = IncompressibleNavierStokes.kinetic_energy(u, setup)
        e = ke[Ip]
        e = fft(e)[ntuple(α -> kx[α] .+ 1, D)...]
        e = abs.(e) ./ size(e, 1)
        e = A * reshape(e, :)
        ehat = max.(e, eps(T)) # Avoid log(0)
        ee = @. abs(ehat - eref) / abs(eref)
        lines!(ax, kint, Array(ee); label)
    end
    axislegend(ax; position = :lt)
    autolimits!(ax)
    fig
end

save("spectrum_error.pdf", fig)
