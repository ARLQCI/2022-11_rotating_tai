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

using QuantumPropagators
using LinearAlgebra
using FFTW

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


include("./rotating_tai.jl")


tlist = collect(range(0, 100ms, step=25ns));
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

phi(t; w0, t_r) = 0.5 * w0 * t - 0.5 * w0* t_r * sin(π * t / t_r) / π

t = collect(range(0, 100ms, length=1000));
plot(t, phi.(t; w0=(2π/sec), t_r=100ms))

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    theta=theta_grid,
    V0=2.2MHz,
    m=N_sites,
    phi=t->phi(t; w0=(2π/sec), t_r=100ms)
);

ϕ = getcontrols(Ĥ)[1]

ϕ

@which evalcontrols(Ĥ.V, IdDict(ϕ => ϕ(0)))

Ĥ₀ = evalcontrols(Ĥ, IdDict(ϕ => ϕ(0))) # XXX
V̂₀ = Ĥ₀.V;

plot(theta_grid./(2π), V̂₀.diag./MHz, xlabel="θ/2π", ylabel="Energy (MHz)")

function get_ground_state(Ĥ₀, theta, θ₀=0.0; steps=10000, d=1.0)
    h = -1im
    Uk2 = exp(-0.5im * h * Ĥ₀.T)
    Uk = exp(-1im * h * Ĥ₀.T)
    Ux = exp(-1im * h * Ĥ₀.V)

    Ψx = convert(Array{ComplexF64}, exp.(-(theta .- θ₀).^2/d^2))
    normalize!(Ψx)

    Ψk = fft(Ψx)

    for i=1:steps
        Ψk = Uk2 * Ψk
        Ψx = ifft(Ψk)
        Ψx = Ux * Ψx
        Ψk = fft(Ψx)
        Ψk = Uk2 * Ψk

        normalize!(Ψk)
    end

    Ψx = ifft(Ψk)
    normalize!(Ψx)

    return Ψx
end

Ψ_ground = get_ground_state(Ĥ₀, theta_grid, π/16,  d=0.05, steps=10_000)

plot(abs2.(Ψ_ground))
