# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.8.0
#     language: julia
#     name: julia-1.8
# ---

using QuantumPropagators
using LinearAlgebra
using FFTW

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


includet("./include/rotating_tai.jl")


includet("./include/split_propagator.jl")

tlist = collect(range(0, 100ms, step=250ns));
#tlist = collect(range(0, 1sec, step=10μs)); # DEBUG
theta_grid = collect(range(0, 2π, length=1024));
#theta_grid = collect(range(0, 2, length=1024)); # DEBUG

length(tlist)

TAI_RADIUS^2 * Rb_mass

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

phi(t; w0, t_r) = 0.5 * w0 * t - 0.5 * w0 * t_r * sin(π * t / t_r) / π

# +
#phi(t; args...) = 0.3

# +
#phi(t; args...) = (0.2π/sec) * t # DEBUG
# -

t = collect(range(0, tlist[end], length=1000));
plot(t ./ sec, phi.(t; w0=(2π / sec), t_r=100ms))

phi(tlist[end]; w0=(2π / sec), t_r=100ms)

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    theta=theta_grid,
    V0=2.2MHz,
    m=N_sites,
    phi=t -> phi(t; w0=(2π / sec), t_r=100ms)
);

ϕ = get_controls(Ĥ)[1]

plot(ϕ.(tlist))

Ĥ₀ = evaluate(Ĥ; vals_dict=IdDict(ϕ => 0.0))
V̂₀ = Ĥ₀.V;

# ## Calculate initial state

plot(theta_grid ./ (2π), V̂₀.diag ./ MHz, xlabel="θ/2π", ylabel="Energy (MHz)")

Ψ_ground = get_ground_state(Ĥ₀, theta_grid, 2π / 16, d=0.05, steps=10_000);

plot(theta_grid ./ (2π), V̂₀.diag ./ MHz, xlabel="θ/2π", ylabel="Energy (MHz)")
_offset = minimum(V̂₀.diag ./ MHz)
plot!(theta_grid ./ (2π), 3 .* abs2.(Ψ_ground) .+ _offset, label="|Ψ₀|²", xlim=(0, 0.15))

# + active=""
# _xshift = 0.0#1
# plot(theta_grid./(2π).-_xshift, abs2.(Ψ_ground), label="|Ψ₀|²", marker=true)
# plot!(theta_grid./(2π), abs2.(Ψ), label="|Ψ|²", marker=true)

# + active=""
# plot(theta_grid./(2π), abs2.(Ψ) .- abs2.(Ψ_ground), label="Δ|Ψ|²")
# -

# ## Propagation (Split Propagator)

tlist = collect(range(0, 100ms, length=50_000));

split_states =
    propagate(Ψ_ground, Ĥ, tlist; method=:splitprop, storage=true, showprogress=true);

function plot_system(generator, states, theta_grid, tlist, n; psi_scale=5)
    t = tlist[n]
    Ĥ = generator
    V = evaluate(Ĥ, t).V.diag
    offset = minimum(V / MHz)
    Ψ = states[:, n]
    fig = plot(theta_grid ./ (2π), V / MHz, xlabel="θ/2π", ylabel="Energy (MHz)", label="V")
    plot!(
        fig,
        theta_grid ./ (2π),
        psi_scale * abs2.(Ψ) .+ offset,
        label="|Ψ|²",
        xlim=(0, 0.15)
    )
    plot!(title="t=$(t/sec)s")
end

anim = @animate for n = 1:1_000:length(tlist)
    plot_system(Ĥ, split_states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)

# ## Optimization Target

Ĥ_tgt = evaluate(Ĥ, tlist[end])
V̂_tgt = Ĥ_tgt.V;

plot(
    theta_grid ./ (2π),
    V̂_tgt.diag ./ MHz,
    xlabel="θ/2π",
    ylabel="Energy (MHz)",
    label="V̂_tgt"
)
plot!(
    theta_grid ./ (2π),
    V̂₀.diag ./ MHz,
    xlabel="θ/2π",
    ylabel="Energy (MHz)",
    label="V̂₀"
)

Ψ_0T = get_ground_state(Ĥ_tgt, theta_grid, 0.1 * 2π, d=0.05, steps=10_000);

Boost = exp.(1im .* (TAI_RADIUS^2 * Rb_mass) .* (2π / sec) .* theta_grid);

Ψ_tgt = Diagonal(Boost) * Ψ_0T

Ψ_T = split_states[:, end];

plot(
    theta_grid ./ (2π),
    V̂_tgt.diag ./ MHz,
    xlabel="θ/2π",
    ylabel="Energy (MHz)",
    label="V(T)"
)
_offset = minimum(V̂_tgt.diag ./ MHz)
plot!(theta_grid ./ (2π), 3 .* abs2.(Ψ_tgt) .+ _offset, label="|Ψtgt|²", xlim=(0, 0.2))
plot!(theta_grid ./ (2π), 3 .* abs2.(Ψ_T) .+ _offset, label="|Ψ(T)|²", xlim=(0, 0.2))

angle(Ψ_tgt ⋅ Ψ_T) / π

abs2(Ψ_tgt ⋅ Ψ_T)

# ## Propagation (Cheby)

tlist = collect(range(0, 100ms, length=1001));

cheby_states = propagate(
    Ψ_ground,
    Ĥ,
    tlist;
    method=:cheby,
    showprogress=true,
    storage=true,
    specrange_method=:arnoldi
);

psi_cheby = cheby_states[:, end];

plot(abs2.(cheby_states[:, end]))

anim = @animate for n = 1:10:length(tlist)
    plot_system(Ĥ, cheby_states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)
