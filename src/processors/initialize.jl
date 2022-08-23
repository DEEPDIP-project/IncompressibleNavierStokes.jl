"""
    initialize!(processor, stepper)

Initialize processor.
"""
function initialize! end

initialize!(logger::Logger, stepper) = logger

# 2D real time plot
function initialize!(plotter::RealTimePlotter, stepper::TimeStepper{M,T,2}) where {M,T}
    (; V, p, t, setup) = stepper
    (; boundary_conditions, grid) = setup
    (; xlims, ylims, x, y, xp, yp) = grid
    (; fieldname, type) = plotter

    if fieldname == :velocity
        up, vp = get_velocity(V, t, setup)
        f = map((u, v) -> √sum(u^2 + v^2), up, vp)
        xf, yf = xp, yp
    elseif fieldname == :vorticity
        if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
            xf = x
            yf = y
        else
            xf = x[2:(end-1)]
            yf = y[2:(end-1)]
        end
        f = get_vorticity(V, t, setup)
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end-1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end-1)]
        end
        f = get_streamfunction(V, t, setup)
    elseif fieldname == :pressure
        error("Not implemented")
        xf, yf = xp, yp
        f = reshape(copy(p), length(xp), length(yp))
    else
        error("Unknown fieldname")
    end

    field = Observable(f)

    fig = Figure()

    if type == heatmap
        lims = Observable(get_lims(f))
        ax, hm = heatmap(fig[1, 1], xf, yf, field; colorrange = lims)
    elseif type ∈ (contour, contourf)
        if ≈(extrema(f)...; rtol = 1e-10)
            μ = mean(f)
            a = μ - 1
            b = μ + 1
            f[1] += 1
            f[end] -= 1
            field[] = f
        else
            a, b = get_lims(f)
        end
        # lims = Observable(LinRange(a, b, 10))
        lims = Observable((a, b))
        ax, hm = type(
            fig[1, 1],
            xf,
            yf,
            field;
            extendlow = :auto,
            extendhigh = :auto,
            levels = @lift(LinRange($(lims)..., 10)),
            colorrange = lims,
        )
    else
        error("Unknown plot type")
    end

    ax.title = titlecase(string(fieldname))
    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    Colorbar(fig[1, 2], hm)
    display(fig)

    @pack! plotter = field, lims

    plotter
end

# 3D real time plot
function initialize!(plotter::RealTimePlotter, stepper::TimeStepper{M,T,3}) where {M,T}
    (; V, p, t, setup) = stepper
    (; grid, boundary_conditions) = setup
    (; x, y, z, xp, yp, zp) = grid
    (; fieldname, type) = plotter

    if fieldname == :velocity
        up, vp, wp = get_velocity(V, t, setup)
        f = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
        xf, yf, zf = xp, yp, zp
    elseif fieldname == :vorticity
        if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
            xf = x
            yf = y
            zf = y
        else
            xf = x[2:(end-1)]
            yf = y[2:(end-1)]
            zf = z[2:(end-1)]
        end
        f = get_vorticity(V, t, setup)
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end-1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end-1)]
        end
        f = get_streamfunction(V, t, setup)
    elseif fieldname == :pressure
        xf, yf, zf = xp, yp, zp
        f = reshape(copy(p), length(xp), length(yp), length(zp))
    else
        error("Unknown fieldname")
    end

    field = Observable(f)

    fig = Figure()

    if type ∈ (contour, contourf)
        type == contour && (type! = contour!)
        type == contourf && (type! = contourf!)
        # lims = Observable(LinRange(get_lims(f, 3)..., 10))
        lims = Observable(get_lims(f, 3))
        ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect = :data)
        hm = type!(
            ax, xf, yf, zf, field;
            levels = @lift(LinRange($(lims)..., 10)),
            colorrange = lims,
            shading = false,
            alpha = 0.05,
        )
    else
        error("Unknown plot type")
    end
    # Colorbar(fig[1, 2], hm; ticks = lims)
    display(fig)

    @pack! plotter = field, lims

    plotter
end

function initialize!(writer::VTKWriter, stepper)
    (; dir, filename) = writer
    ispath(dir) || mkpath(dir)
    pvd = paraview_collection(joinpath(dir, filename))
    @pack! writer = pvd
    writer
end

function initialize!(tracer::QuantityTracer, stepper)
    tracer.t = zeros(0)
    tracer.maxdiv = zeros(0)
    tracer.umom = zeros(0)
    tracer.vmom = zeros(0)
    tracer.wmom = zeros(0)
    tracer.k = zeros(0)
    tracer
end
