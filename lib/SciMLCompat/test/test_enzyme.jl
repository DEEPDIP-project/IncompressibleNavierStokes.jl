using IncompressibleNavierStokes
using SciMLCompat
using KernelAbstractions
using Enzyme
Enzyme.API.runtimeActivity!(true)

TIME_TOL = 1.5
N_REPETITIONS = 1000

# Test the Enzyme implementation
T = Float32
ArrayType = Array
Re = T(1_000)
n = 32
N = n + 2
lims = T(0), T(1);
x, y = LinRange(lims..., n + 1), LinRange(lims..., n + 1);
setup = Setup(x, y; Re, ArrayType);
_backend = get_backend(rand(Float32, 10))

###### BC_U
myapply_bc_u! = _get_enz_bc_u!(_backend, setup)

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local A = (rand(Float32, N, N), rand(Float32, N, N))
    A0 = (copy(A[1]), copy(A[2]))
    B = (copy(A[1]), copy(A[2]))
    IncompressibleNavierStokes.apply_bc_u!(A, T(0), setup)
    myapply_bc_u!(B)
    @test A[1] == B[1]
    @test A[2] == B[2]
    @test A[1] != A0[1]
    @test A[2] != A0[2]
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local A = (rand(Float32, N, N), rand(Float32, N, N))
    IncompressibleNavierStokes.apply_bc_u!(A, 0.0f0, setup)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local A = (rand(Float32, N, N), rand(Float32, N, N))
    myapply_bc_u!(A)
end
# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme bc_u too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme bc_u too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if it is differentiable
A = (rand(Float32, N, N), rand(Float32, N, N))
dA = Enzyme.make_zero(A)
Enzyme.autodiff(Enzyme.Reverse, myapply_bc_u!, Const, DuplicatedNoNeed(A, dA))

####### BC_P
myapply_bc_p! = _get_enz_bc_p!(_backend, setup);

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local A = rand(Float32, N, N)
    A0 = copy(A)
    B = copy(A)
    IncompressibleNavierStokes.apply_bc_p!(A, T(0), setup)
    myapply_bc_p!(B)
    @test A == B
    @test A != A0
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local A = rand(Float32, N, N)
    IncompressibleNavierStokes.apply_bc_p!(A, 0.0f0, setup)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local A = rand(Float32, N, N)
    myapply_bc_p!(A)
end
# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme bc_p too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme bc_p too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if it is differentiable
A = rand(Float32, N, N);
dA = Enzyme.make_zero(A);
Enzyme.autodiff(Enzyme.Reverse, myapply_bc_p!, Const, DuplicatedNoNeed(A, dA))

####### Momentum
my_momentum! = _get_enz_momentum!(_backend, nothing, setup)

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local F = random_field(setup, T(0))
    u0 = copy.(u)
    F0 = copy.(F)
    IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
    my_momentum!(F0, u, T(0))
    @test F == F0
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local F = random_field(setup, T(0))
    IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local F = random_field(setup, T(0))
    my_momentum!(F, u, T(0))
end

# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme momentum too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme momentum too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if it is differentiable
u = random_field(setup, T(0))
F = random_field(setup, T(0))
du = Enzyme.make_zero(u)
dF = Enzyme.make_zero(F)
Enzyme.autodiff(
    Enzyme.Reverse,
    my_momentum!,
    Const,
    DuplicatedNoNeed(F, dF),
    DuplicatedNoNeed(u, du),
    Const(T(0)),
)

####### Divergence
my_divergence! = _get_enz_div!(_backend, setup)

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local d = rand(T, (N, N))
    local u = random_field(setup, T(0))
    d0 = copy(d)
    IncompressibleNavierStokes.divergence!(d, u, setup)
    local z = Enzyme.make_zero(d)
    my_divergence!(d0, u, z)
    @test d == d0
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local d = rand(T, (N, N))
    local u = random_field(setup, T(0))
    IncompressibleNavierStokes.divergence!(d, u, setup)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local d = rand(T, (N, N))
    local u = random_field(setup, T(0))
    local z = Enzyme.make_zero(d)
    my_divergence!(d, u, z)
end

# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme divergence too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme divergence too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if it is differentiable
d = rand(T, (N, N))
dd = Enzyme.make_zero(d)
u = random_field(setup, T(0))
du = Enzyme.make_zero(u)
z = Enzyme.make_zero(d)
dz = Enzyme.make_zero(z)
Enzyme.autodiff(
    Enzyme.Reverse,
    my_divergence!,
    Const,
    DuplicatedNoNeed(d, dd),
    DuplicatedNoNeed(u, du),
    DuplicatedNoNeed(z, dz),
)

####### Pressure solver
my_psolver! = _get_enz_psolver!(setup)
INSpsolver! = IncompressibleNavierStokes.psolver_direct(setup);

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local p = rand(T, (N, N))
    local d = rand(T, (N, N))
    local ft = rand(T, n * n + 1)
    local pt = rand(T, n * n + 1)
    p0 = copy(p)
    INSpsolver!(p, d)
    my_psolver!(p0, d, ft, pt)
    @test isapprox(p, p0; rtol=1e-6, atol=1e-6)
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local p = rand(T, (N, N))
    local d = rand(T, (N, N))
    INSpsolver!(p, d)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local p = rand(T, (N, N))
    local d = rand(T, (N, N))
    local ft = rand(T, n * n + 1)
    local pt = rand(T, n * n + 1)
    my_psolver!(p, d, ft, pt)
end

# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme psolver too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme psolver too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if it is differentiable
p = rand(T, (N, N));
d = rand(T, (N, N));
ft = rand(T, n * n + 1);
pt = rand(T, n * n + 1);
dp = Enzyme.make_zero(p);
dd = Enzyme.make_zero(d);
dft = Enzyme.make_zero(ft);
dpt = Enzyme.make_zero(pt);
Enzyme.autodiff(
    Enzyme.Reverse,
    my_psolver!,
    Const,
    DuplicatedNoNeed(p, dp),
    DuplicatedNoNeed(d, dd),
    DuplicatedNoNeed(ft, dft),
    DuplicatedNoNeed(pt, dpt),
)

####### applypressure
my_applypressure! = _get_enz_applypressure!(_backend, setup);

# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < TIME_TOL * time_INS "enzyme applypressure too slow: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < TIME_TOL * allocation_INS "enzyme applypressure too much memory: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if the implementation is correct
for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local p = rand(T, (N, N))
    u0 = copy.(u)
    IncompressibleNavierStokes.applypressure!(u, p, setup)
    my_applypressure!(u0, p)
    @test u == u0
end

# Speed test
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local p = rand(T, (N, N))
    IncompressibleNavierStokes.applypressure!(u, p, setup)
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i = 1:N_REPETITIONS
    local u = random_field(setup, T(0))
    local p = rand(T, (N, N))
    my_applypressure!(u, p)
end

# Check if it is differentiable
u = random_field(setup, T(0))
p = rand(T, (N, N))
du = Enzyme.make_zero(u)
dp = Enzyme.make_zero(p)
Enzyme.autodiff(
    Enzyme.Reverse,
    my_applypressure!,
    Const,
    DuplicatedNoNeed(u, du),
    DuplicatedNoNeed(p, dp),
)