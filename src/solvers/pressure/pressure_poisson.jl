"""
    pressure_poisson(solver, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Non-mutating/allocating/out-of-place version.

See also [`pressure_poisson!`](@ref).
"""
function pressure_poisson end

function pressure_poisson(solver::DirectPressureSolver, f)
    # Assume the Laplace matrix is known (A) and is possibly factorized

    # Use pre-determined decomposition
    solver.A_fact \ f
end

function pressure_poisson(solver::CGPressureSolver, f)
    (; A, abstol, reltol, maxiter) = solver
    cg(A, f; abstol, reltol, maxiter)
end

function pressure_poisson(solver::SpectralPressureSolver, f)
    (; Ahat) = solver

    f = reshape(f, size(Ahat))

    # Fourier transform of right hand side
    fhat = fft(f)

    # Solve for coefficients in Fourier space
    phat = @. -fhat / Ahat

    # Transform back
    p = ifft(phat)

    reshape(real.(p), :)
end

"""
    pressure_poisson!(solver, p, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Mutating/non-allocating/in-place version.

See also [`pressure_poisson`](@ref).
"""
function pressure_poisson! end

function pressure_poisson!(solver::DirectPressureSolver, p, f)
    # Assume the Laplace matrix is known (A) and is possibly factorized

    f = view(f, :)
    p = view(p, :)

    # Use pre-determined decomposition
    p .= solver.A_fact \ f
end

function pressure_poisson!(solver::CGPressureSolver, p, f)
    (; A, abstol, reltol, maxiter) = solver
    f = view(f, :)
    p = view(p, :)
    cg!(p, A, f; abstol, reltol, maxiter)
end

function pressure_poisson!(solver::SpectralPressureSolver, p, f)
    (; Ahat, fhat, phat) = solver

    phat .= complex.(f)

    # Fourier transform of right hand side
    fft!(phat)

    # Solve for coefficients in Fourier space
    @. phat = -phat / Ahat

    # Transform back
    ifft!(phat)
    @. p = real(phat)

    p
end
