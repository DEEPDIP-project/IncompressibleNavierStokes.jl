include("load_results.jl")

## final u : from Syver
ufinal = load("$savepath/ufinal.jld2")["ufinal"];

GLMakie.activate!()
GLMakie.with_theme(; fontsize = 25, palette) do
    ## Reference box for eddy comparison
    x1 = 0.3
    x2 = 0.5
    y1 = 0.5
    y2 = 0.7
    box = [
        GLMakie.Point2f(x1, y1),
        GLMakie.Point2f(x2, y1),
        GLMakie.Point2f(x2, y2),
        GLMakie.Point2f(x1, y2),
        GLMakie.Point2f(x1, y1),
    ]
    path = "$plotdir/les_fields/$mname"
    ispath(path) || mkpath(path)
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        setup = setups_test[igrid]
        name = "$path/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid)"
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        function makeplot(u, title, suffix)
            fig = fieldplot(
                (; u, t = T(0));
                setup,
                title,
                docolorbar = false,
                size = (500, 500),
            )
            GLMakie.lines!(box; linewidth = 5, color = Cycled(2)) # Red in palette
            fname = "$(name)_$(suffix).png"
            save(fname, fig)
            ## run(`convert $fname -trim $fname`) # Requires imagemagick
        end
        iorder == 2 &&
            GLMakie.makeplot(ufinal.u_ref[igrid, ifil], "Reference, $fil, $nles", "ref")
        iorder == 2 && makeplot(
            ufinal.u_nomodel[igrid, ifil],
            "No closure, $fil, $nles",
            "nomodel",
        )
        makeplot(
            ufinal.u_smag[igrid, ifil, iorder],
            "Smagorinsky, $lesmodel, $fil, $nles",
            "smag",
        )
        makeplot(
            ufinal.u_cnn_prior[igrid, ifil, iorder],
            "CNN (prior), $lesmodel, $fil, $nles",
            "prior",
        )
        makeplot(
            ufinal.u_cnn_post[igrid, ifil, iorder],
            "CNN (post), $lesmodel, $fil, $nles",
            "post",
        )
    end
end

## u trajectory
(; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post) = load("$outdir/$mname/u_output.jld2", "u_output");
# u_ref and u_nomodel have the format [n_grids, n_filters], u_ref has an extra time-element: initial condition.
# u_cnn_prior, u_cnn_post and u_smag have the format [n_grids, n_filters, n_order] 
# Order is only > 1 for posteriori and smagorinsky.

# Visualizations
function visualize(field, setup; save_path = nothing)
    GLMakie.activate!()
    u = field[1]
    o = GLMakie.Observable((; u, temp = nothing, t = nothing))
    # default field is vorticity but can be changed to velocitynorm or 1. 
    scene = fieldplot(o; setup,
        #fieldname = :velocitynorm,
        #fieldname = 1,
    )
    if save_path !== nothing
        GLMakie.record(scene, save_path, 1:length(field); framerate = 30) do i
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
field = u_smag[3, 1, 1]
visualize(field, setup; save_path="$plotdir/smag_1.gif")
field = u_smag[3, 1, 2]
visualize(field, setup; save_path="$plotdir/smag_2.gif")

# Error fields
field = [(a .- b) for (a, b) in zip(u_nomodel[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/nomodel_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_prior[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/prior_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_post[3, 1, 1], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/post_1_error.mp4")
field = [(a .- b) for (a, b) in zip(u_cnn_post[3, 1, 2], u_ref[3, 1][2:end])]
visualize(field, setup; save_path="$plotdir/post_2_error.mp4")