# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Julia 1.7.2
#     language: julia
#     name: julia-1.7
# ---

using DrWatson
#quickactivate("/SSD/SSD/Dropbox/GITHUB_PROJECTS/JuliaQuantumControl/JuliaQuantumControl/QuantumControlBase.jl/test")

using QuantumPropagators
using LinearAlgebra
using FFTW
FFTW.set_provider!("mkl")

using Revise

using Plots

const μm = 1
const μs = 1
const ns = 1e-3μs
const cm = 1e4μm
const met = 1e6μm
const sec = 1e6μs
const ms = 1e3μs
const MHz = 2π
const Dalton = 1.5746097504353806e+01
const Rb_mass = 86.91Dalton;
const TAI_RADIUS = 42μm
const N_sites = 8;

includet("./rotating_tai.jl")

includet("./split_propagator.jl")

tlist = collect(range(0, 1sec, step=10μs));
theta_grid = collect(range(0, 2π, length=2048));

function rotating_tai_hamiltonian(; tlist, V0, m, theta, phi, mass=(TAI_RADIUS^2 * Rb_mass))
    V = RotTAI_PotentialGenerator(; V0, m=m, theta=theta, phi)
    dθ = theta[2] - theta[1]
    n_theta = length(theta)
    pgrid = 2π * fftfreq(n_theta, 1 / dθ)
    K = Diagonal(pgrid .^ 2 / (2 * mass))
    Ψrand = Array{ComplexF64}(undef, n_theta)
    fft_op = plan_fft!(Ψrand)
    ifft_op = plan_ifft!(Ψrand)
    SplitGenerator(K, V, Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)
end

# +
function phi(t; w0, t_r, t_loop) 
    if t <= t_r
        return 0.5 * w0 * t - 0.5 * w0 * t_r * sin(π * t / t_r) / π
    elseif t <= t_r + t_loop
        return w0 * (t - t_r) + phi(t_r; w0, t_r, t_loop)
    else
        t̄ = t_r + t_loop
        return 0.5 * w0 * t_r + phi(t - t̄ - t_r; w0, t_r, t_loop) + phi(t̄; w0, t_r, t_loop)
    end
end

phi(1sec; w0=(2π/sec), t_r=100ms, t_loop=800ms)
# -

t = collect(range(0, tlist[end], length=1000));
α = (2π)/phi(1sec; w0=(2π/sec), t_r=100ms, t_loop=800ms) # So it ends in 2π
plot(t ./ sec, phi.(t; w0=α*(2π/sec), t_r=100ms, t_loop=800ms) / π, label="", ylabel="ϕ/π")

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    theta=theta_grid,
    V0=2.2MHz,
    m=N_sites,
    phi=t->phi(t; w0=(2π/sec), t_r=100ms, t_loop=800ms)
);

ϕ = getcontrols(Ĥ)[1]
Ĥ₀ = evalcontrols(Ĥ, IdDict(ϕ => 0.0))
V̂₀ = Ĥ₀.V;
Ψ_ground = get_ground_state(Ĥ₀, theta_grid, 2π/16,  d=0.05, steps=10_000);

function plot_system(generator, states, theta_grid, tlist, n; psi_scale=5)
    t = tlist[n]
    Ĥ = generator
    V = evalcontrols(Ĥ, IdDict(getcontrols(Ĥ)[1] => getcontrols(Ĥ)[1](t))).V.diag
    offset = minimum(V/MHz)
    Ψ = states[:,n]
    fig = plot(theta_grid./(2π), V/MHz, xlabel="θ/2π", ylabel="Energy (MHz)", label="V")
    plot!(fig, theta_grid./(2π), psi_scale*abs2.(Ψ).+offset, label="|Ψ|²")#, xlim=(0, 0.15))
    plot!(title="t=$(t/sec)s")
end

states = propagate(Ψ_ground, Ĥ, tlist; method=:splitprop, storage=true, showprogress=true);

anim = @animate for n=1:length(tlist)÷200:length(tlist)
    plot_system(Ĥ, states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)

# ## Full scheme

# +
function test(; dir=1, Ω=0)
    tlist = collect(range(0, 0.4sec, step=25μs));
    theta_grid = collect(range(0, 2π, length=4096));
    
    α = (2π)/phi(1sec; w0=(2π/sec), t_r=100ms, t_loop=800ms) 
    
    Ĥ = rotating_tai_hamiltonian(
        tlist=tlist,
        theta=theta_grid,
        V0=2.2MHz,
        m=N_sites,
        #phi=t -> Ω * t
        phi=t -> dir * phi(t; w0=α*(2π/sec), t_r=100ms, t_loop=800ms) + Ω * t
    );
    
    ϕ = getcontrols(Ĥ)[1]
    Ĥ₀ = evalcontrols(Ĥ, IdDict(ϕ => 0.0))
    V̂₀ = Ĥ₀.V;
    
    Ψ_ground = get_ground_state(Ĥ₀, theta_grid, 2π/16,  d=0.05, steps=10_000)
    U_boost = exp.(+1im * (TAI_RADIUS^2 * Rb_mass) * Ω * theta_grid)
    moving_Ψ_ground = U_boost .* Ψ_ground
    
    states = propagate(Ψ_ground, Ĥ, tlist; method=:splitprop, storage=true, showprogress=true);
    
    anim = @animate for n=1:length(tlist)÷200:length(tlist)
        plot_system(Ĥ, states, theta_grid, tlist, n; psi_scale=20)
    end
    anim
end

Ω = 1/sec

anim = test(;dir=-1, Ω=Ω)
gif(anim, "anim.gif", fps=10)
# -

function prop_squeme(; dir=1, Ω=0)
    tlist = collect(range(0, 1sec, step=50μs));
    theta_grid = collect(range(0, 2π, length=4096));
    #theta_grid = collect(range(0, 2π, length=2048));
    
    α = (2π)/phi(1sec; w0=(2π/sec), t_r=100ms, t_loop=800ms) 
    
    Ĥ = rotating_tai_hamiltonian(
        tlist=tlist,
        theta=theta_grid,
        V0=1.1MHz,
        m=N_sites,
        #phi=t -> Ω * t
        phi=t -> dir * phi(t; w0=α*(2π/sec), t_r=100ms, t_loop=800ms) + Ω * t
    );
    
    ϕ = getcontrols(Ĥ)[1]
    Ĥ₀ = evalcontrols(Ĥ, IdDict(ϕ => 0.0))
    V̂₀ = Ĥ₀.V;
    
    Ψ_ground = get_ground_state(Ĥ₀, theta_grid, 2π/16,  d=0.05, steps=10_000)
    U_boost = exp.(+1im * (TAI_RADIUS^2 * Rb_mass) * Ω * theta_grid)
    moving_Ψ_ground = U_boost .* Ψ_ground
    
    propagate(moving_Ψ_ground, Ĥ, tlist; method=:splitprop, showprogress=true);
end

# +
function eval_scheme(Ω)
    Ψ0 = prop_squeme(;dir=1, Ω=Ω)
    Ψ1 = prop_squeme(;dir=-1, Ω=Ω)
    
    ## Maybe the overlap with the target first?
    # π/2-pulse
    Ψup = (Ψ0 + Ψ1)
    abs2(Ψup ⋅ Ψup)/16
end

#(1 + exp(ia)) * (1 + exp(-ia)) = 1 + 2 cos(a) + 1
# -

Ω_list = collect(range(0, 0.5/sec, length=41))
P_list = eval_scheme.(Ω_list)

plot(Ω_list*sec, P_list, label="", xlabel="Ω (rad/s)", ylabel="P")


