function project(u, setup; psolver)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    div = divergence(u, setup)
    div = @. div * Ω

    # Solve the Poisson equation
    p = poisson(psolver, div)

    # Apply pressure correction term
    p = apply_bc_p(p, T(0), setup)
    G = pressuregradient(p, setup)
    u .- G
end

function project!(u, setup; psolver, div, p)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    divergence!(div, u, setup)
    @. div *= Ω

    # Solve the Poisson equation
    poisson!(psolver, p, div)
    apply_bc_p!(p, T(0), setup)

    # Apply pressure correction term
    applypressure!(u, p, setup)
end
