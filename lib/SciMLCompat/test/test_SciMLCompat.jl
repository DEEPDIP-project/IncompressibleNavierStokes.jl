using IncompressibleNavierStokes
using SciMLCompat
using Lux
using ComponentArrays
using Random
rng = Random.default_rng();
using Enzyme
Enzyme.API.runtimeActivity!(true)

# Tests to check the compatibility with Enzyme and the SciML ecosystem

# Define the problem
T = Float32
ArrayType = Array
Re = T(1_000)
n = 32
N = n + 2
lims = T(0), T(1);
x, y = LinRange(lims..., n + 1), LinRange(lims..., n + 1);
setup = Setup(x, y; Re, ArrayType);
_backend = get_backend(rand(Float32, 10))

# Create the right hand side function
F = create_right_hand_side_enzyme(_backend, setup, T, n)

# define the variables
u0 = zeros(T, (N, N, 2))
du = similar(u0)

# test that the force is working
F(du, u0, nothing, T(0))
@test sum(du) == 0
u = rand(T, (N, N, 2))
F(du, u, nothing, T(0))
@test sum(du) != 0

# define and compile a mock a-priori loss
function apriori(u_ini, temp)
    F(temp, u_ini, nothing, 0.0f0)
    return sum(u0 - temp)
end
du = Enzyme.make_zero(u);
temp = similar(u);
dtemp = Enzyme.make_zero(temp);
apriori(u, temp)

# check that the a-priori function is differentiable
Enzyme.autodiff(
    Enzyme.Reverse,
    apriori,
    Active,
    DuplicatedNoNeed(u, du),
    DuplicatedNoNeed(temp, dtemp),
)
# and that there is no gradient since there are no trainable parameters
@test sum(dtemp) == 0

# Define a simple convolutional neural network
dummy_NN = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    x -> view(x, :),
)
θ, st = Lux.setup(rng, dummy_NN)
θ = ComponentArray(θ)
Lux.apply(dummy_NN, u, θ, st)[1];

# Define the right hand side function with the neural network closure   
dudt_nn(du, u, θ, t) = begin
    F(du, u, nothing, t)
    view(du, :) .= view(du, :) .+ Lux.apply(dummy_NN, u, θ, st)[1]
    nothing
end

# test that the force is working
dudt_nn(du, u0, θ, T(0))
@test sum(du) == 0
u = rand(T, (N, N, 2))
dudt_nn(du, u, θ, T(0))
@test sum(du) != 0

# define and compile a mock a-priori loss with the neural network
function apriori_nn(u_ini, p, temp)
    dudt_nn(temp, u_ini, p, 0.0f0)
    return sum(u0 - temp)
end
du = Enzyme.make_zero(u);
dθ = Enzyme.make_zero(θ);
temp = similar(u);
dtemp = Enzyme.make_zero(temp);
apriori_nn(u, θ, temp)

# check that the a-priori function is differentiable
Enzyme.autodiff(
    Enzyme.Reverse,
    apriori_nn,
    Active,
    DuplicatedNoNeed(u, du),
    DuplicatedNoNeed(θ, dθ),
    DuplicatedNoNeed(temp, dtemp),
)
# and check that the gradient is not zero
@test sum(dθ) != 0

###### Test the compatibility with the SciML ecosystem
using DifferentialEquations
using SciMLSensitivity
dt = T(1e-3);
trange = [T(0), T(2) * dt]
saveat = [dt, 2dt];
u = stack(random_field(setup, T(0)))
prob = ODEProblem{true}(dudt_nn, u, trange; p = θ)
ode_data = Array(solve(prob, RK4(); u0 = u, p = θ, saveat = saveat))
ode_data += T(0.1) * rand(Float32, size(ode_data))

# define the loss function
function loss(
    l::Vector{Float32},
    θ,
    u0::Array{Float32},
    tspan::Vector{Float32},
    t::Vector{Float32},
)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, θ)
    pred = Array(solve(myprob, RK4(); u0 = u0, p = θ, saveat = t))
    l .= Float32(sum(abs2, ode_data - pred))
    nothing
end
l = [T(0.0)];
loss(l, θ, u, trange, saveat);

# and test that the loss is differentiable
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+ T(1);
dθ = Enzyme.make_zero(θ);
du = Enzyme.make_zero(u);
Enzyme.autodiff(
    Enzyme.Reverse,
    loss,
    DuplicatedNoNeed(l, dl),
    DuplicatedNoNeed(θ, dθ),
    DuplicatedNoNeed(u, du),
    Const(trange),
    Const(saveat),
)
# check the gradient
@test sum(dθ) != 0

#### Test the compatibility with the Optimisation ecosystem
using Optimization
using OptimizationOptimisers
using Optimisers

extra_par = [u, trange, saveat, du, dθ];
function loss_gradient(G, θ, extra_par)
    u0, trange, saveat, du0, dθ = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dθ)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(
        Enzyme.Reverse,
        loss,
        DuplicatedNoNeed([T(0)], [T(1)]),
        DuplicatedNoNeed(θ, dθ),
        DuplicatedNoNeed(u0, du0),
        Const(trange),
        Const(saveat),
    )
    # The gradient matters only for theta
    G .= dθ
    nothing
end

# Trigger the gradient
G = copy(dθ);
oo = loss_gradient(G, θ, extra_par)

# This is to call loss using only P
function over_loss(θ)
    loss(l, θ, u, trange, saveat)
    return l
end
callback = function (θ, l; doplot = false)
    println(l)
    return false
end
callback(θ, over_loss(θ))

optf = Optimization.OptimizationFunction(
    (p, _) -> over_loss(p);
    grad = (G, p, e) -> loss_gradient(G, p, e),
)
optprob = Optimization.OptimizationProblem(optf, θ, extra_par)

result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 10,
)