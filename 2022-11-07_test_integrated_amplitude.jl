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
#     display_name: Julia 1.8.2
#     language: julia
#     name: julia-1.8
# ---

# # Rotating TAI with ω(t)

# Here, we use $\omega(t)$ as the control field and $\phi(t) = \int \omega(t) dt$ as the control amplitude.

using QuantumPropagators
using LinearAlgebra
using FFTW

import QuantumControl.Controls: discretize, discretize_on_midpoints, evalcontrols

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


includet("./integrated_amplitude.jl")

includet("./rotating_tai.jl")


includet("./split_propagator.jl")

tlist = collect(range(0, 100ms, length=50_001));
theta_grid = collect(range(0, 2π, length=1024));

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

omega(t; w0, t_r) = w0 * sin(π*t/(2t_r))^2;

phi(; w0, t_r) = IntegratedAmplitude(
    discretize_on_midpoints(
        t->omega(t; w0, t_r),
        tlist
    )
);

function evaluate(generator::SplitGenerator, tlist, n)
    vals_dict = IdDict(c => c[n] for c ∈ getcontrols(generator))
    evalcontrols(generator, vals_dict, tlist, n)
end

function evaluate(ampl::IntegratedAmplitude, tlist, n)
    vals_dict = IdDict(ampl.control => ampl.control[n])
    return evalcontrols(ampl, vals_dict, tlist, n)
end

function discretize_on_midpoints(ampl::IntegratedAmplitude, tlist)
    N = length(tlist) - 1
    return [evaluate(ampl, tlist, n) for n ∈ 1:N]
end

function discretize(ampl::IntegratedAmplitude, tlist)
    vals = discretize_on_midpoints(ampl, tlist)
    return discretize(vals, tlist)
end

plot(tlist ./ sec, discretize(phi(; w0=(2π/sec), t_r=100ms), tlist); legend=:topleft, label="ϕ(t)", xlabel="time")

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    theta=theta_grid,
    V0=2.2MHz,
    m=N_sites,
    phi=phi(; w0=(2π/sec), t_r=100ms)
);

ω = getcontrols(Ĥ)[1];

getcontrols(Ĥ)[1] ≡ ω

Ĥ₀ = evaluate(Ĥ, tlist, 1)
V̂₀ = Ĥ₀.V;

# ## Calculate initial state

plot(theta_grid./(2π), V̂₀.diag./MHz, xlabel="θ/2π", ylabel="Energy (MHz)")

Ψ_ground = get_ground_state(Ĥ₀, theta_grid, 2π/16,  d=0.05, steps=10_000);

plot(theta_grid./(2π), V̂₀.diag./MHz, xlabel="θ/2π", ylabel="Energy (MHz)")
_offset = minimum(V̂₀.diag./MHz)
plot!(theta_grid./(2π), 3 .* abs2.(Ψ_ground).+_offset, label="|Ψ₀|²", xlim=(0,0.15))

# ## Propagation (Split Propagator)

split_states = propagate(Ψ_ground, Ĥ, tlist; method=:splitprop, storage=true, showprogress=true);

function plot_system(generator, states, theta_grid, tlist, n; psi_scale=5)
    t = tlist[n]
    Ĥ = generator
    V = evaluate(Ĥ, tlist, min(n, length(tlist)-1)).V.diag
    offset = minimum(V/MHz)
    Ψ = states[:,n]
    fig = plot(theta_grid./(2π), V/MHz, xlabel="θ/2π", ylabel="Energy (MHz)", label="V")
    plot!(fig, theta_grid./(2π), psi_scale*abs2.(Ψ).+offset, label="|Ψ|²", xlim=(0, 0.15))
    plot!(title="t=$(t/sec)s")
end

anim = @animate for n=1:500:length(tlist)
    plot_system(Ĥ, split_states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)
