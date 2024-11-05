using TestItemRunner

@testmodule Setup2D begin
    name="2D"
    using IncompressibleNavierStokes
    T = Float64
    Re = T(1_000)
    n = 4
    lims = T(0), T(1)
    x = tanh_grid(lims..., n), tanh_grid(lims..., n, 1.3)
    bc = DirichletBC(), DirichletBC()
    setup = Setup(; x, Re, boundary_conditions = (bc, bc))
    uref(dim, x, y, args...) = -(dim == 1) * sin(x) * cos(y) + (dim == 2) * cos(x) * sin(y)
    u_ins = velocityfield(setup, uref, T(0))
    u_io = cat(u_ins...; dims = ndims(u_ins[1]) + 1)
end

@testmodule Setup3D begin
    name="3D"
    using IncompressibleNavierStokes
    T = Float64
    Re = T(1_000)
    n = 4
    lims = T(0), T(1)
    x = tanh_grid(lims..., n, 1.2), tanh_grid(lims..., n, 1.1), cosine_grid(lims..., n)
    bc = DirichletBC(), DirichletBC(), DirichletBC()
    setup = Setup(; x, Re, boundary_conditions = (bc, bc, bc))
    uref(dim, x, y, args...) = -(dim == 1) * sin(x) * cos(y) + (dim == 2) * cos(x) * sin(y)
    u_ins = velocityfield(setup, uref, T(0))
    u_io = cat(u_ins...; dims = ndims(u_ins[1]) + 1)

end

@testitem "Divergence" setup = [Setup2D, Setup3D] begin
    for setup in (Setup2D,Setup3D)
        divergence(setup.u_io, setup.setup) # warm cache
        div_io, io_time = @timed divergence(setup.u_io, setup.setup)
      
        divergence(setup.u_ins, setup.setup)
        div_ins, ins_time = @timed divergence(setup.u_ins, setup.setup)
        name = setup.name

        if ins_time < io_time
            @warn("$name Divergence: IO Array took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test all(!isnan, div_io)
        @test div_io == div_ins
    end
end

@testitem "Pressure gradient" setup = [Setup2D, Setup3D] begin
    using Random
    for setup_ in (Setup2D, Setup3D)
        setup = setup_.setup
        (; Iu, Ip, dimension) = setup.grid
        D = dimension()
        v_ins = randn!.(vectorfield(setup))
        v_io = cat(v_ins...; dims = ndims(v_ins[1]) + 1)
        p = randn!(scalarfield(setup))
        T = eltype(p)


        apply_bc_u(v_ins, T(0), setup)
        v_ins, ins_time = @timed apply_bc_u(v_ins, T(0), setup)
        apply_bc_u(v_io, T(0), setup)
        v_io, io_time = @timed apply_bc_u(v_io, T(0), setup)

        name = setup_.name
        if ins_time < io_time
            @warn("$name apply_bc: IO Array took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test v_io == cat(v_ins...; dims = ndims(v_ins[1]) + 1)

        p = apply_bc_p(p, T(0), setup)
        Dv = divergence(v_io, setup)
        
        pressuregradient_io(p, setup)
        Gp_io::Array, io_time = @timed pressuregradient_io(p, setup)
        pressuregradient(p, setup)
        Gp_ins::Tuple, ins_time = @timed pressuregradient(p, setup)

        if ins_time < io_time
            @warn("$name pressuregradient: IO Array took more time than INS Tuple ($io_time,$ins_time)")
        end

        @test Gp_io == cat(Gp_ins...; dims = ndims(Gp_ins[1]) + 1)

        ΩDv = scalewithvolume(Dv, setup)
        pDv = sum((p.*ΩDv)[Ip])
        vGp = if D == 2
            vGpx = v_io[:,:,1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* Gp_io[:,:,1]
            vGpy = v_io[:,:,2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* Gp_io[:,:,2]
            sum(vGpx[Iu[1]]) + sum(vGpy[Iu[2]])
        elseif D == 3
            vGpx =
                v_io[:,:,:,1] .* setup.grid.Δu[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                reshape(setup.grid.Δ[3], 1, 1, :) .* Gp_io[:,:,:,1]
            vGpy =
                v_io[:,:,:,2] .* setup.grid.Δ[1] .* reshape(setup.grid.Δu[2], 1, :) .*
                reshape(setup.grid.Δ[3], 1, 1, :) .* Gp_io[:,:,:,2]
            vGpz =
                v_io[:,:,:,3] .* setup.grid.Δ[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                reshape(setup.grid.Δu[3], 1, 1, :) .* Gp_io[:,:,:,3]
            sum(vGpx[Iu[1]]) + sum(vGpy[Iu[2]]) + sum(vGpz[Iu[3]])
        end
        @test Gp_io isa Array
        @test pDv ≈ -vGp # Check that D = -G'
    end
end

#=

Victor
@testitem "Convection" setup = [Setup2D, Setup3D] begin
    for (u, setup) in ((Setup2D.u, Setup2D.setup), (Setup3D.u, Setup3D.setup))
        (; Iu, Δ, Δu) = setup.grid
        T = eltype(u[1])
        c = convection(u, setup)
        D = length(u)
        uCu = if D == 2
            uCux = u[1] .* Δu[1] .* Δ[2]' .* c[1]
            uCuy = u[2] .* Δ[1] .* Δu[2]' .* c[2]
            sum(uCux[Iu[1]]) + sum(uCuy[Iu[2]])
        elseif D == 3
            Δu1, Δu2, Δu3 = Δu[1], reshape(Δu[2], 1, :), reshape(Δu[3], 1, 1, :)
            Δp1, Δp2, Δp3 = Δ[1], reshape(Δ[2], 1, :), reshape(Δ[3], 1, 1, :)
            uCux = @. u[1] * Δu1 * Δp2 * Δp3 .* c[1]
            uCuy = @. u[2] * Δp1 * Δu2 * Δp3 .* c[2]
            uCuz = @. u[3] * Δp1 * Δp2 * Δu3 .* c[3]
            sum(uCux[Iu[1]]) + sum(uCuy[Iu[2]]) + sum(uCuz[Iu[3]])
        end
        @test c isa Tuple
        @test c[1] isa Array{T}
        @test uCu ≈ 0 atol = 1e-12 # Check skew-symmetry
    end
end
 =#

@testitem "Diffusion" setup = [Setup2D, Setup3D] begin
    for (u_ins, u_io, setup, name) in ((Setup2D.u_ins, Setup2D.u_io, Setup2D.setup, Setup2D.name), (Setup3D.u_ins, Setup3D.u_io, Setup3D.setup, Setup3D.name))
        T = eltype(u_ins[1])
        (; Iu, Δ, Δu) = setup.grid
        diffusion(u_io, setup)
        d_io, io_time = @timed diffusion(u_io, setup)

        diffusion(u_ins, setup)
        d_ins, ins_time = @timed diffusion(u_ins, setup)

        if ins_time > io_time
            @warn("$name IO Array Diffusion took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test all(!isnan, d_io)
        @test d_io == stack(d_ins)
    end
end

@testitem "Convection-Diffusion" setup = [Setup2D, Setup3D] begin
    for (u_ins, u_io, setup, name) in ((Setup2D.u_ins, Setup2D.u_io, Setup2D.setup, Setup2D.name), (Setup3D.u_ins, Setup3D.u_io, Setup3D.setup, Setup3D.name))
        IncompressibleNavierStokes.convectiondiffusion!(zero.(u_ins), u_ins, setup)
        cd_ins, ins_time = @timed IncompressibleNavierStokes.convectiondiffusion!(zero.(u_ins), u_ins, setup)

        cd_io, io_time = @timed IncompressibleNavierStokes.convectiondiffusion!(zero.(u_io), u_io, setup)

        if ins_time > io_time
            @warn("$name IO Array Convection-Diffusion took more time than INS Tuple ($io_time,$ins_time)")
        end

        c_io = convection(u_io, setup)
        d_io = diffusion(u_io, setup)
        @test all(cd_io .≈ c_io .+ d_io)
        @test cd_io == stack(cd_ins)
    end
end


@testitem "Momentum" setup = [Setup2D, Setup3D] begin
    for (u_ins, u_io, setup, name) in ((Setup2D.u_ins, Setup2D.u_io, Setup2D.setup, Setup2D.name), (Setup3D.u_ins, Setup3D.u_io, Setup3D.setup, Setup3D.name))
        T = eltype(u_ins[1])
        momentum(u_ins, nothing, T(1), setup)
        m_ins, ins_time = @timed momentum(u_ins, nothing, T(1), setup)

        momentum(u_io, nothing, T(1), setup)
        m_io, io_time = @timed momentum(u_io, nothing, T(1), setup)

        if ins_time > io_time
            @warn("$name IO Array Momentum took more time than INS Tuple ($io_time,$ins_time)")
        end

        @test all(!isnan, m_io)
        @test m_io == stack(m_ins)
    end
end

@testitem "Other fields" setup = [Setup2D, Setup3D] begin
    using Random
    for (u_ins, u_io, setup, name) in ((Setup2D.u_ins, Setup2D.u_io, Setup2D.setup, Setup2D.name), (Setup3D.u_ins, Setup3D.u_io, Setup3D.setup, Setup3D.name))
        T = eltype(u_ins[1])
        D = length(u_ins)

        vorticity(u_ins, setup)
        ω_ins, ins_time = @timed vorticity(u_ins, setup)
        vorticity(u_io, setup)
        ω_io, io_time = @timed vorticity(u_io, setup)
        if ins_time > io_time
            @warn("$name IO Array Vorticity took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test all(!isnan, stack(ω_io))
        @test ω_io == ω_ins

        smagorinsky_closure(setup)(u_ins, 0.1)
        s_ins, ins_time = @timed smagorinsky_closure(setup)(u_ins, 0.1)
        smagorinsky_closure(setup)(u_io, 0.1)
        s_io, io_time = @timed smagorinsky_closure(setup)(u_io, 0.1)
        if ins_time > io_time
            @warn("$name IO Array Smagorinsky Closure took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test all(!isnan, stack(s_io))
        @test s_io == s_ins

        # test nabla before tensorbasis

        using Base.Iterators: flatten
        tensorbasis(u_ins, setup) isa Tuple
        tensorbasis(u_io, setup) isa Tuple
        tb_ins, ins_time = @timed tensorbasis(u_ins, setup)
        tb_io, io_time = @timed tensorbasis(u_io, setup)
        if ins_time > io_time
            @warn("$name IO Array Tensor Basis took more time than INS Tuple ($io_time,$ins_time)")
        end
        @test typeof(tb_ins) == typeof(tb_io)
        for i in [1,2]
            for j in eachindex(tb_ins[i])
                for k in eachindex(tb_ins[i][j])
                    @assert maximum(abs.(tb_io[i][j][k] .- tb_ins[i][j][k])) < 1e-8 "Failed at $i,$j,$k: $(tb_io[i][j][k]) vs $(tb_ins[i][j][k])"
                end
            end
        end
        #for i in eachindex(tb_ins)
        #    @test tb_io[i] == tb_ins[i]
        #end
#        println(sum(tb_io[1] .- tb_ins[1]))
#        println(typeof(collect(view(flatten(tb_io) .- flatten(tb_ins),:))))
#        println(map(maximum, collect(view(flatten(tb_io) .- flatten(tb_ins),:))))

#        @test interpolate_u_p(u, setup) isa Tuple
#        D == 2 && @test interpolate_ω_p(ω, setup) isa Array{T}
#        D == 3 && @test interpolate_ω_p(ω, setup) isa Tuple
#        @test Dfield(p, setup) isa Array{T}
#        @test Qfield(u, setup) isa Array{T}
#        D == 2 && @test_throws AssertionError eig2field(u, setup)
#        D == 3 && @test eig2field(u, setup) isa Array{T} broken = D == 3
#        @test kinetic_energy(u, setup) isa Array{T}
#        @test total_kinetic_energy(u, setup) isa T
#        @test dissipation_from_strain(u, setup) isa Array{T}
#        @test get_scale_numbers(u, setup) isa NamedTuple
    end
end
