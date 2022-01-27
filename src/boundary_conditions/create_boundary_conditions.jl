"""
    create_boundary_conditions(u_bc, v_bc[, w_bc]; T = Float64, bc_unsteady, bc_type, kwargs...)

Create discrete boundary condtions.

Values should either be scalars or vectors. All values `(u, v, p, k, e)` are defined at x, y locations,
i.e. the corners of pressure volumes, so they cover the entire domain, including corners.
"""
function create_boundary_conditions end

# 2D version
function create_boundary_conditions(u_bc, v_bc; T = Float64, bc_unsteady, bc_type, kwargs...)
    bc_type.u.x[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-left")
    bc_type.u.x[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-right")
    bc_type.u.y[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-low")
    bc_type.u.y[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-up")

    bc_type.v.x[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-left")
    bc_type.v.x[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-right")
    bc_type.v.y[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-low")
    bc_type.v.y[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-up")

    # Pressure (for boundaries marked with `:pressure`)
    p∞ = 0
    p_bc = (; x = (p∞, p∞), y = (p∞, p∞))

    # K-eps values
    k_bc = (; x = (0, 0), y = (0, 0))
    e_bc = (; x = (0, 0), y = (0, 0))

    BC{T}(; bc_unsteady, bc_type..., u_bc, v_bc, p_bc, k_bc, e_bc, kwargs...)
end

# 3D version
function create_boundary_conditions(u_bc, v_bc, w_bc; T = Float64, bc_unsteady, bc_type, kwargs...)
    bc_type.u.x[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-left")
    bc_type.u.x[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-right")
    bc_type.u.y[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-low")
    bc_type.u.y[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-up")
    bc_type.u.z[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-back")
    bc_type.u.z[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-front")

    bc_type.v.x[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-left")
    bc_type.v.x[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-right")
    bc_type.v.y[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-low")
    bc_type.v.y[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-up")
    bc_type.v.z[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-back")
    bc_type.v.z[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-front")

    bc_type.w.x[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for w-left")
    bc_type.w.x[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for w-right")
    bc_type.w.y[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for w-low")
    bc_type.w.y[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for w-up")
    bc_type.w.z[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for w-back")
    bc_type.w.z[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for w-front")

    # Pressure (for boundaries marked with `:pressure`)
    p∞ = 0
    p_bc = (; x = (p∞, p∞), y = (p∞, p∞), z = (p∞, p∞))

    # K-eps values
    k_bc = (; x = (0, 0), y = (0, 0), z = (0, 0))
    e_bc = (; x = (0, 0), y = (0, 0), z = (0, 0))

    BC{T}(; bc_unsteady, bc_type..., u_bc, v_bc, w_bc, p_bc, k_bc, e_bc, kwargs...)
end
