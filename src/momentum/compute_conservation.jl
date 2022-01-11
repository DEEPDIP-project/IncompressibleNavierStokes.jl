"""
    compute_conservation(V, t, setup)

Compute mass, momentum and energy conservation properties of velocity field.
"""
function compute_conservation end

# 2D version
function compute_conservation(V, t, setup::Setup{T,2}) where {T}
    (; M, yM) = setup.operators
    (; indu, indv, Ω, x, y, xp, yp, hx, hy, gx, gy) = setup.grid
    (; u_bc, v_bc) = setup.bc

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]

    setup.bc.bc_unsteady && set_bc_vectors!(setup, t)
    uLe_i = reshape(u_bc.(x[1], yp, t, [setup])   , :)
    uRi_i = reshape(u_bc.(x[end], yp, t, [setup]) , :)
    vLo_i = reshape(v_bc.(xp, y[1], t, [setup])   , :)
    vUp_i = reshape(v_bc.(xp, y[end], t, [setup]) , :)

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)

    # Add boundary contributions in case of Dirichlet BC
    setup.bc.u.x[1] == :dirichlet && (umom += sum(uLe_i .* hy) * gx[1])
    setup.bc.u.x[2] == :dirichlet && (umom += sum(uRi_i .* hy) * gx[end])
    setup.bc.v.y[1] == :dirichlet && (vmom += sum(vLo_i .* hx) * gy[1])
    setup.bc.v.y[2] == :dirichlet && (vmom += sum(vUp_i .* hx) * gy[end])

    # Calculate total kinetic energy
    k = 1 / 2 * sum(Ω .* V .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    setup.bc.u.x[1] == :dirichlet && (k += 1 / 2 * sum(uLe_i .^ 2 .* hy) * gx[1])
    setup.bc.u.x[2] == :dirichlet && (k += 1 / 2 * sum(uRi_i .^ 2 .* hy) * gx[end])
    setup.bc.v.y[1] == :dirichlet && (k += 1 / 2 * sum(vLo_i .^ 2 .* hx) * gy[1])
    setup.bc.v.y[2] == :dirichlet && (k += 1 / 2 * sum(vUp_i .^ 2 .* hx) * gy[end])

    maxdiv, umom, vmom, k
end

# 3D version
function compute_conservation(V, t, setup::Setup{T,3}) where {T}
    (; M, yM) = setup.operators
    (; indu, indv, indw, Ω, x, y, z, xp, yp, zp, hx, hy, hz, gx, gy, gz) = setup.grid
    (; u_bc, v_bc, w_bc) = setup.bc

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]
    Ωw = @view Ω[indw]

    setup.bc.bc_unsteady && set_bc_vectors!(setup, t)
    uLe_i = reshape(u_bc.(x[1], yp, zp', t, [setup])   , :)
    uRi_i = reshape(u_bc.(x[end], yp, zp', t, [setup]) , :)
    vLo_i = reshape(v_bc.(xp, y[1], zp', t, [setup])   , :)
    vUp_i = reshape(v_bc.(xp, y[end], zp', t, [setup]) , :)
    wBa_i = reshape(w_bc.(xp, yp', z[1], t, [setup])   , :)
    wFr_i = reshape(w_bc.(xp, yp', z[end], t, [setup]) , :)

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)
    wmom = sum(Ωw .* wₕ)

    # Add boundary contributions in case of Dirichlet BC
    setup.bc.u.x[1] == :dirichlet && (umom += sum(uLe_i .* (hz ⊗ hy)) * gx[1])
    setup.bc.u.x[2] == :dirichlet && (umom += sum(uRi_i .* (hz ⊗ hy)) * gx[end])
    setup.bc.v.y[1] == :dirichlet && (vmom += sum(vLo_i .* (hz ⊗ hx)) * gy[1])
    setup.bc.v.y[2] == :dirichlet && (vmom += sum(vUp_i .* (hz ⊗ hx)) * gy[end])
    setup.bc.w.z[1] == :dirichlet && (wmom += sum(wBa_i .* (hy ⊗ hx)) * gz[1])
    setup.bc.w.z[2] == :dirichlet && (wmom += sum(wFr_i .* (hy ⊗ hx)) * gz[end])

    # Calculate total kinetic energy
    k = 1 / 2 * sum(Ω .* V .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    setup.bc.u.x[1] == :dirichlet && (k += 1 / 2 * sum(uLe_i .^ 2 .* (hz ⊗ hy)) * gx[1])
    setup.bc.u.x[2] == :dirichlet && (k += 1 / 2 * sum(uRi_i .^ 2 .* (hz ⊗ hy)) * gx[end])
    setup.bc.v.y[1] == :dirichlet && (k += 1 / 2 * sum(vLo_i .^ 2 .* (hz ⊗ hx)) * gy[1])
    setup.bc.v.y[2] == :dirichlet && (k += 1 / 2 * sum(vUp_i .^ 2 .* (hz ⊗ hx)) * gy[end])
    setup.bc.w.z[1] == :dirichlet && (k += 1 / 2 * sum(wBa_i .^ 2 .* (hy ⊗ hx)) * gz[1])
    setup.bc.w.z[2] == :dirichlet && (k += 1 / 2 * sum(wFr_i .^ 2 .* (hy ⊗ hx)) * gz[end])

    maxdiv, umom, vmom, wmom, k
end