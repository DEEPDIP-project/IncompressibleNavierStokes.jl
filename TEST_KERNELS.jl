using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

# Setup
Re = 2e3
n = 128
x = LinRange(0.0, 1.0, n + 1), LinRange(0.0, 1.0, n + 1);
setup = INS.Setup(; x=x, Re=Re);
ustart = INS.random_field(setup, 0.0);
psolver = INS.psolver_spectral(setup);

create_right_hand_side_tuple(setup, psolver) = function right_hand_side(u, p, t)
    u = eachslice(u; dims = ndims(u))
    u = (u...,)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
    stack(PF)
end
create_right_hand_side_array(setup, psolver) = function right_hand_side(u, p, t)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
end

# Note: Requires `stack(u)` to create one array
f_tuple = create_right_hand_side_tuple(setup, psolver)
f_tuple(stack(ustart), nothing, 0.0)

f_array = create_right_hand_side_array(setup, psolver)
f_array(stack(ustart), nothing, 0.0)

@assert f_tuple(stack(ustart), nothing, 0.0) ≈ f_array(ustart, nothing, 0.0)

