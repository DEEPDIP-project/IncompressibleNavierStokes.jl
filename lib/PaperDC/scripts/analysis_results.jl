using CairoMakie: CairoMakie
include("load_results.jl")

# 1. Errors: 
# 1.1 a-priori
eprior = let
    prior = zeros(T, 3, 2)
    post = zeros(T, 3, 2, 2)
    for ig = 1:3, ifil = 1:2
        println("ig = $ig, ifil = $ifil")
        testset = device(io_test[ig, ifil])
        err = create_relerr_prior(closure, testset...)
        prior[ig, ifil] = err(θ_cnn_prior[ig, ifil])
        for iorder = 1:2
            post[ig, ifil, iorder] = err(θ_cnn_post[ig, ifil, iorder])
        end
    end
    (; prior, post)
end
clean()
eprior.prior |> x -> reshape(x, :) |> x -> round.(x; digits = 2)
eprior.post |> x -> reshape(x, :, 2) |> x -> round.(x; digits = 2)

# plot
CairoMakie.activate!()
fig = CairoMakie.with_theme(; palette) do
    nles = [n[1] for n in params_test.nles][1:3]
    ifil = 1
    fig = CairoMakie.Figure(; size = (500, 400))
    ax = CairoMakie.Axis(
        fig[1, 1];
        xscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        title = "Relative a-priori error $(ifil == 1 ? " (FA)" : " (VA)")",
    )
    linestyle = :solid
    label = "No closure"
    CairoMakie.scatterlines!(
        nles,
        ones(T, length(nles));
        color = CairoMakie.Cycled(1),
        linestyle,
        marker = :circle,
        label,
    )
    label = "CNN (Lprior)"
    CairoMakie.scatterlines!(
        nles,
        eprior.prior[:, ifil];
        color = CairoMakie.Cycled(2),
        linestyle,
        marker = :utriangle,
        label,
    )
    label = "CNN (Lpost, DIF)"
    CairoMakie.scatterlines!(
        nles,
        eprior.post[:, ifil, 1];
        color = CairoMakie.Cycled(3),
        linestyle,
        marker = :rect,
        label,
    )
    label = "CNN (Lpost, DCF)"
    CairoMakie.scatterlines!(
        nles,
        eprior.post[:, ifil, 2];
        color = CairoMakie.Cycled(4),
        linestyle,
        marker = :diamond,
        label,
    )
    CairoMakie.axislegend(; position = :lb)
    CairoMakie.ylims!(ax, (T(-0.05), T(1.05)))
    name = "$plotdir/convergence"
    ispath(name) || mkpath(name)
    CairoMakie.save("$name/$(mname)_prior_ifilter$ifil.pdf", fig)
    fig
end

# 1.2 a-posteriori
data_test = load("$outdir/data/data_test.jld2", "data_test");
data_train = load("$outdir/data/data_train.jld2", "data_train");


(; e_nm, e_smag, e_cnn, e_cnn_post) = let
    e_nm = zeros(T, size(data_test.data)...)
    e_smag = zeros(T, size(data_test.data)..., 2)
    e_cnn = zeros(T, size(data_test.data)..., 2)
    e_cnn_post = zeros(T, size(data_test.data)..., 2)
    for iorder = 1:2, ifil = 1:2, ig = 1:size(data_test.data, 1)
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        projectorder = getorder(iorder)
        setup = setups_test[ig]
        psolver = psolver_spectral(setup)
        data = (; u = device.(data_test.data[ig, ifil].u), t = data_test.t)
        nupdate = 2
        ## No model
        ## Only for closurefirst, since projectfirst is the same
        if iorder == 2
            err = create_relerr_post(; data, setup, psolver, closure_model = nothing, nupdate)
            e_nm[ig, ifil] = err(nothing)
        end
        ## Smagorinsky
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(RK44(; T), getorder(iorder)),
            closure_model = IncompressibleNavierStokes.smagorinsky_closure(setup),
            nupdate,
        )
        e_smag[ig, ifil, iorder] = err(θ_smag[ifil, iorder])
        ## CNN
        ## Only the first grids are trained for
        if ig ≤ size(setups_train, 1)
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                method = RKProject(RK44(; T), getorder(iorder)),
                closure_model = wrappedclosure(closure, setup),
                nupdate,
            )
            e_cnn[ig, ifil, iorder] = err(θ_cnn_prior[ig, ifil])
            e_cnn_post[ig, ifil, iorder] = err(θ_cnn_post[ig, ifil, iorder])
        end
    end
    (; e_nm, e_smag, e_cnn, e_cnn_post)
end
clean()
round.(
    [e_nm[:] reshape(e_smag, :, 2) reshape(e_cnn, :, 2) reshape(e_cnn_post, :, 2)][
        [1:3; 6:8], :,];
    sigdigits = 2,
)

# Plot
CairoMakie.activate!()
CairoMakie.with_theme(; palette) do
    iorder = 2
    lesmodel = iorder == 1 ? "DIF" : "DCF"
    ntrain = size(setups_train, 1)
    nles = [n[1] for n in params_test.nles][1:ntrain]
    fig = CairoMakie.Figure(; size = (500, 400))
    ax = CairoMakie.Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        title = "Relative error ($lesmodel)",
    )
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "No closure"
        ifil == 2 && (label = nothing)
        CairoMakie.scatterlines!(
            nles,
            e_nm[1:ntrain, ifil];
            color = Cycled(1),
            linestyle,
            marker = :circle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "Smagorinsky"
        ifil == 2 && (label = nothing)
        CairoMakie.scatterlines!(
            nles,
            e_smag[1:ntrain, ifil, iorder];
            color = Cycled(2),
            linestyle,
            marker = :utriangle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN (prior)"
        ifil == 2 && (label = nothing)
        CairoMakie.scatterlines!(
            nles[1:ntrain],
            e_cnn[1:ntrain, ifil, iorder];
            color = Cycled(3),
            linestyle,
            marker = :rect,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN (post)"
        ifil == 2 && (label = nothing)
        CairoMakie.scatterlines!(
            nles[1:ntrain],
            e_cnn_post[1:ntrain, ifil, iorder];
            color = Cycled(4),
            linestyle,
            marker = :diamond,
            label,
        )
    end
    CairoMakie.axislegend(; position = :lb)
    CairoMakie.ylims!(ax, (T(0.025), T(1.00)))
    name = "$plotdir/convergence"
    ispath(name) || mkpath(name)
    save("$name/$(mname)_iorder$iorder.pdf", fig)
    fig
end

# 2. Kinetic energy evolution
kineticenergy = let
    clean()
    ngrid, nfilter = size(io_train)
    ke_ref = fill(zeros(T, 0), ngrid, nfilter)
    ke_nomodel = fill(zeros(T, 0), ngrid, nfilter)
    ke_smag = fill(zeros(T, 0), ngrid, nfilter, 2)
    ke_cnn_prior = fill(zeros(T, 0), ngrid, nfilter, 2)
    ke_cnn_post = fill(zeros(T, 0), ngrid, nfilter, 2)
    for iorder = 1:2, ifil = 1:nfilter, ig = 1:ngrid
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_test[ig]
        psolver = psolver_spectral(setup)
        t = data_test.t
        ustart = data_test.data[ig, ifil].u[1]
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
        ewriter = processor() do state
            ehist = zeros(T, 0)
            on(state) do (; u, n)
                if n % nupdate == 0
                    e = IncompressibleNavierStokes.total_kinetic_energy(u, setup)
                    push!(ehist, e)
                end
            end
            state[] = state[] # Compute initial energy
            ehist
        end
        processors = (; ewriter)
        if iorder == 1
            ## Does not depend on projection order
            ke_ref[ig, ifil] = map(
                u -> IncompressibleNavierStokes.total_kinetic_energy(u, setup),
                data_test.data[ig, ifil].u,
            )
            ke_nomodel[ig, ifil] =
                solve_unsteady(; setup, ustart, tlims, Δt, processors, psolver)[2].ewriter
        end
        ke_smag[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = IncompressibleNavierStokes.smagorinsky_closure(setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_smag[ifil, iorder],
            )[2].ewriter
        ke_cnn_prior[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_cnn_prior[ig, ifil],
            )[2].ewriter
        ke_cnn_post[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_cnn_post[ig, ifil, iorder],
            )[2].ewriter
    end
    (; ke_ref, ke_nomodel, ke_smag, ke_cnn_prior, ke_cnn_post)
end;
clean();

# plot
CairoMakie.activate!()
CairoMakie.with_theme(; palette) do
    t = data_test.t
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        fig = CairoMakie.Figure(; size = (500, 400))
        ax = CairoMakie.Axis(
            fig[1, 1];
            xlabel = "t",
            ylabel = "E(t)",
            title = "Kinetic energy: $lesmodel, $fil",
        )
        CairoMakie.lines!(
            ax,
            t,
            kineticenergy.ke_ref[igrid, ifil];
            color = CairoMakie.Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        CairoMakie.lines!(
            ax,
            t,
            kineticenergy.ke_nomodel[igrid, ifil];
            color = CairoMakie.Cycled(1),
            label = "No closure",
        )
        CairoMakie.lines!(
            ax,
            t,
            kineticenergy.ke_smag[igrid, ifil, iorder];
            color = CairoMakie.Cycled(2),
            label = "Smagorinsky",
        )
        CairoMakie.lines!(
            ax,
            t,
            kineticenergy.ke_cnn_prior[igrid, ifil, iorder];
            color = CairoMakie.Cycled(3),
            label = "CNN (prior)",
        )
        CairoMakie.lines!(
            ax,
            t,
            kineticenergy.ke_cnn_post[igrid, ifil, iorder];
            color = CairoMakie.Cycled(4),
            label = "CNN (post)",
        )
        iorder == 1 && CairoMakie.axislegend(; position = :lt)
        iorder == 2 && CairoMakie.axislegend(; position = :lb)
        name = "$plotdir/energy_evolution/$mname/"
        ispath(name) || mkpath(name)
        CairoMakie.save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end

# 3. Divergence evolution
divs = let
    clean()
    ngrid, nfilter = size(io_train)
    d_ref = fill(zeros(T, 0), ngrid, nfilter)
    d_nomodel = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_smag = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_cnn_prior = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_cnn_post = fill(zeros(T, 0), ngrid, nfilter, 3)
    for iorder = 1:3, ifil = 1:nfilter, ig = 1:ngrid
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_test[ig]
        psolver = psolver_spectral(setup)
        t = data_test.t
        ustart = data_test.data[ig, ifil].u[1]
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
        dwriter = processor() do state
            div = fill!(similar(setup.grid.x[1], setup.grid.N), 0)
            dhist = zeros(T, 0)
            on(state) do (; u, n)
                if n % nupdate == 0
                    IncompressibleNavierStokes.divergence!(div, u, setup)
                    d = view(div, setup.grid.Ip)
                    d = sum(abs2, d) / length(d)
                    d = sqrt(d)
                    push!(dhist, d)
                end
            end
            state[] = state[] # Compute initial divergence
            dhist
        end
        if iorder == 1
            ## Does not depend on projection order
            d_ref[ig, ifil] = map(data_test.data[ig, ifil].u) do u
                div = IncompressibleNavierStokes.divergence(u, setup)
                d = view(div, setup.grid.Ip)
                d = sum(abs2, d) / length(d)
                d = sqrt(d)
            end
        end
        s(closure_model, θ) =
            solve_unsteady(;
                (; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(RK44(; T), getorder(iorder)),
                Δt,
                processors = (; dwriter),
                psolver,
                θ,
            )[2].dwriter
        iorder_use = iorder == 3 ? 2 : iorder
        d_nomodel[ig, ifil, iorder] = s(nothing, nothing)
        d_smag[ig, ifil, iorder] =
            s(IncompressibleNavierStokes.smagorinsky_closure(setup), θ_smag[ifil, iorder_use])
        d_cnn_prior[ig, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_prior[ig, ifil])
        d_cnn_post[ig, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_post[ig, ifil, iorder_use])
    end
    (; d_ref, d_nomodel, d_smag, d_cnn_prior, d_cnn_post)
end;
clean();
# Check that divergence is within reasonable bounds
divs.d_ref .|> extrema
divs.d_nomodel .|> extrema
divs.d_smag .|> extrema
divs.d_cnn_prior .|> extrema
divs.d_cnn_post .|> extrema

# plot
CairoMakie.activate!()
CairoMakie.with_theme(; palette) do
    t = data_test.t
    # for islog in (true, false)
    for islog in (false,)
        for iorder = 1:2, ifil = 1:2, igrid = 1:3
            println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
            lesmodel = if iorder == 1
                "DIF"
            elseif iorder == 2
                "DCF"
            elseif iorder == 3
                "DCF-RHS"
            end
            fil = ifil == 1 ? "FA" : "VA"
            nles = params_test.nles[igrid]
            fig = CairoMakie.Figure(; size = (500, 400))
            ax = CairoMakie.Axis(
                fig[1, 1];
                yscale = islog ? log10 : identity,
                xlabel = "t",
                title = "Divergence: $lesmodel, $fil,  $nles",
            )
            CairoMakie.lines!(ax, t, divs.d_nomodel[igrid, ifil, iorder]; label = "No closure")
            CairoMakie.lines!(ax, t, divs.d_smag[igrid, ifil, iorder]; label = "Smagorinsky")
            CairoMakie.lines!(ax, t, divs.d_cnn_prior[igrid, ifil, iorder]; label = "CNN (prior)")
            CairoMakie.lines!(ax, t, divs.d_cnn_post[igrid, ifil, iorder]; label = "CNN (post)")
            CairoMakie.lines!(
                ax,
                t,
                divs.d_ref[igrid, ifil];
                color = CairoMakie.Cycled(1),
                linestyle = :dash,
                label = "Reference",
            )
            iorder == 2 && ifil == 1 && axislegend(; position = :rt)
            islog && CairoMakie.ylims!(ax, (T(1e-6), T(1e3)))
            name = "$plotdir/divergence/$mname/$(islog ? "log" : "lin")"
            ispath(name) || mkpath(name)
            CairoMakie.save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
        end
    end
end

# 4. Kinetic energy spectra
CairoMakie.activate!()
fig = with_theme(; palette) do
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        setup = setups_test[igrid]
        fields =
            [
                ufinal.u_ref[igrid, ifil],
                ufinal.u_nomodel[igrid, ifil],
                ufinal.u_smag[igrid, ifil, iorder],
                ufinal.u_cnn_prior[igrid, ifil, iorder],
                ufinal.u_cnn_post[igrid, ifil, iorder],
            ]
        (; Ip) = setup.grid
        (; A, κ, K) = IncompressibleNavierStokes.spectral_stuff(setup)
        specs = map(fields) do u
            up = u
            e = sum(up) do u
                u = u[Ip]
                uhat = fft(u)[ntuple(α -> 1:K[α], 2)...]
                abs2.(uhat) ./ (2 * prod(size(u))^2)
            end
            e = A * reshape(e, :)
            ## e = max.(e, eps(T)) # Avoid log(0)
            ehat = Array(e)
        end
        kmax = maximum(κ)
        ## Build inertial slope above energy
        krange = [T(16), T(κ[end])]
        slope, slopelabel = -T(3), L"$\kappa^{-3}"
        slopeconst = maximum(specs[1] ./ κ .^ slope)
        offset = 3
        inertia = offset .* slopeconst .* krange .^ slope
        ## Nice ticks
        logmax = round(Int, log2(kmax + 1))
        xticks = T(2) .^ (0:logmax)
        ## Make plot
        fig = CairoMakie.Figure(; size = (500, 400))
        ax = CairoMakie.Axis(
            fig[1, 1];
            xticks,
            xlabel = "κ",
            xscale = log10,
            yscale = log10,
            limits = (1, kmax, T(1e-8), T(1)),
            title = "Kinetic energy: $lesmodel, $fil",
        )
        CairoMakie.lines!(ax, κ, specs[2]; color = CairoMakie.Cycled(1), label = "No model")
        CairoMakie.lines!(ax, κ, specs[3]; color = CairoMakie.Cycled(2), label = "Smagorinsky")
        CairoMakie.lines!(ax, κ, specs[4]; color = CairoMakie.Cycled(3), label = "CNN (prior)")
        CairoMakie.lines!(ax, κ, specs[5]; color = CairoMakie.Cycled(4), label = "CNN (post)")
        CairoMakie.lines!(ax, κ, specs[1]; color = CairoMakie.Cycled(1), linestyle = :dash, label = "Reference")
        CairoMakie.lines!(ax, krange, inertia; color = CairoMakie.Cycled(1), label = slopelabel, linestyle = :dot)
        CairoMakie.axislegend(ax; position = :cb)
        CairoMakie.autolimits!(ax)
        CairoMakie.ylims!(ax, (T(1e-3), T(0.35)))
        name = "$plotdir/energy_spectra/$mname"
        ispath(name) || mkpath(name)
        CairoMakie.save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end
clean();