# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Julia 1.8.0
#     language: julia
#     name: julia-1.8
# ---

# # Rotating TAI with ω(t) - Fast Splitting

# Here, we use $\omega(t)$ as the control field and $\phi(t) = \int \omega(t) dt$ as the control amplitude.

# ## Hamiltonian

using QuantumPropagators
using LinearAlgebra
using FFTW

import QuantumControl.Controls: discretize, discretize_on_midpoints, evaluate

using Revise

using Plots

const μm = 1;
const μs = 1;
const ns = 1e-3μs;
const cm = 1e4μm;
const met = 1e6μm;
const sec = 1e6μs;
const ms = 1e3μs;
const MHz = 2π;
const Dalton = 1.5746097504353806e+01;


const RUBIDIUM_MASS = 86.91Dalton;
const TAI_RADIUS = 42μm
const N_SITES = 8;
const SEPARATION_TIME = 1ms;
const OMEGA_TARGET = (1000π / sec);
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;

includet("./include/integrated_amplitude.jl")

includet("./include/rotating_tai.jl")


includet("./include/split_propagator.jl")

tlist = collect(range(0, SEPARATION_TIME, length=(Int(SEPARATION_TIME ÷ ms) * 1000 + 1)));
theta_grid = collect(range(0, 0.75π, length=6144));

typeof(theta_grid)

# **TODO: choose number of time grid points so that cheby is stable and split-prop gives the same result**

length(tlist)

println(
    "Storage for full Hamiltonian matrix: $(sizeof(zeros(Float64, length(theta_grid), length(theta_grid))) / 1024^2) MB"
)

function rotating_tai_hamiltonian(;
    tlist,
    theta,
    phi,
    direction=1,
    Omega=0.0,
    V0=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)
    V = RotTAI_PotentialGenerator(; V0, m, theta, phi, direction, Omega)
    dθ = theta[2] - theta[1]
    n_theta = length(theta)
    pgrid = 2π * fftfreq(n_theta, 1 / dθ)
    K = Diagonal(pgrid .^ 2 / (2 * mass))
    Ψrand = Array{ComplexF64}(undef, n_theta)
    fft_op = plan_fft!(Ψrand)
    ifft_op = plan_ifft!(Ψrand)
    SplitGenerator(K, V, Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)
end

omega(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

phi(; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) =
    IntegratedAmplitude(discretize_on_midpoints(t -> omega(t; w0, t_r), tlist));

function discretize_on_midpoints(ampl::IntegratedAmplitude, tlist)
    N = length(tlist) - 1
    return [evaluate(ampl, tlist, n) for n ∈ 1:N]
end

function discretize(ampl::IntegratedAmplitude, tlist)
    vals = discretize_on_midpoints(ampl, tlist)
    return discretize(vals, tlist)
end

plot(
    tlist ./ sec,
    discretize(omega.(tlist), tlist) / (2π / sec);
    legend=:topleft,
    label="ω(t)",
    xlabel="time (sec)",
    ylabel="angular velocity (2π/sec)"
)

plot(
    tlist ./ ms,
    discretize(phi(), tlist) ./ π;
    legend=:topleft,
    label="ϕ(t)",
    xlabel="time (ms)",
    ylabel="phase / π"
)

Ĥ = rotating_tai_hamiltonian(tlist=tlist, theta=theta_grid, phi=phi());

typeof(Ĥ)

typeof(Ĥ.V)

typeof(Ĥ.V.phi)

fieldnames(typeof(Ĥ.V))

Ĥ₀ = evaluate(Ĥ, tlist, 1)
V̂₀ = Ĥ₀.V;

typeof(Ĥ₀)

typeof(Ĥ₀.V)

# ## Calculate initial state

Ψ₀ = get_ground_state(Ĥ₀, theta_grid, π / 8, d=0.05, steps=10_000);

plot(theta_grid ./ π, V̂₀.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀.diag ./ MHz)
plot!(
    theta_grid ./ π,
    50 .* abs2.(Ψ₀) .+ _offset,
    label="|Ψ₀|²",
    xlim=(0, 0.3),
    legend=:right
)

println("⟨θ⟩_Ψ₀ = $(round(abs(dot(Ψ₀, Diagonal(theta_grid), Ψ₀) / π), digits=10))π")

# ## Diagonalization

function cut_grid(
    Ψ::Vector{ComplexF64},
    theta_grid::Vector{Float64};
    θmin::Float64,
    θmax::Float64
)
    filter = (theta_grid .≥ θmin) .& (theta_grid .≤ θmax)
    return Ψ[filter], theta_grid[filter]
end

_, theta_cut = cut_grid(Ψ₀, theta_grid; θmin=0.1π, θmax=0.15π);

Ĥ_cut = evaluate(
    rotating_tai_hamiltonian(tlist=tlist, theta=theta_cut, V0=2.2MHz, m=N_SITES, phi=phi()),
    tlist,
    1
);

function kinetic_matrix(theta_grid::Vector{Float64}, mass::Float64)
    nx::Int64 = length(theta_grid)
    dx::Float64 = theta_grid[2] - theta_grid[1]
    D = zeros(Float64, nx, nx)
    @inbounds for i = 1:nx
        for j = 1:nx
            if i ≠ j
                D[j, i] = ((-1)^(j - i)) / sin(π * (j - i) / nx)
            end
        end
    end
    T = D^2
    lmul!(-π^2 / (2 * mass * nx^2 * dx^2), T)
    return Hermitian(T)
end

function diagonalize_operator(op::SplitOperator, theta_grid, mass=EFFECTIVE_MASS)
    T = kinetic_matrix(theta_grid, mass)
    V = op.V
    H = Hermitian(T + V)
    return eigen(H)
end

eigensys = diagonalize_operator(Ĥ_cut, theta_cut);

function plot_eigensys(H, eigensys; n_max=10, scale=2, kwargs...)
    fig = plot(theta_cut ./ π, H.V.diag ./ MHz)
    for (n, λₙ) ∈ enumerate(eigensys.values)
        (n ≤ n_max) || break
        ϕₙ = eigensys.vectors[:, n]
        plot!(fig, theta_cut ./ π, scale .* abs2.(ϕₙ) .+ (λₙ / MHz))
    end
    plot!(fig; xlabel="θ/π", ylabel="Energy (MHz)", legend=false, kwargs...)
    return fig
end

plot_eigensys(Ĥ_cut, eigensys; xlim=(0.115, 0.135), ylim=(-2.2, -2.1))

plot(eigensys.values ./ MHz, ylabel="Energy (MHz)", xlabel="eigenvalue index", legend=false)

# ## Target state

Ĥ_tgt = evaluate(Ĥ, tlist, length(tlist) - 1);

V̂_tgt = Ĥ_tgt.V;

Boost = Diagonal(exp.(1im .* EFFECTIVE_MASS .* OMEGA_TARGET .* theta_grid));

Ψ_tgt = Boost * get_ground_state(Ĥ_tgt, theta_grid, 0.6π, d=0.05, steps=10_000);

plot(theta_grid ./ π, V̂_tgt.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_tgt.diag ./ MHz)
plot!(theta_grid ./ π, 50 .* abs2.(Ψ_tgt) .+ _offset, label="|Ψ_tgt|²")

# ## Propagation (Split Propagator)

split_states = propagate(
    Ψ₀,
    Ĥ,
    tlist;
    method=:cheby,
    specrange_method=:arnoldi,
    storage=true,
    showprogress=true
);

function plot_system(generator, states, theta_grid, tlist, n; psi_scale=50)
    t = tlist[n]
    Ĥ = generator
    V = evaluate(Ĥ, tlist, min(n, length(tlist) - 1)).V.diag
    offset = minimum(V / MHz)
    Ψ = states[:, n]
    fig = plot(theta_grid ./ π, V / MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
    plot!(fig, theta_grid ./ π, psi_scale * abs2.(Ψ) .+ offset, label="|Ψ|²")
    plot!(title="t=$(t/ms)ms")
end

anim = @animate for n = 1:(length(tlist)÷100):length(tlist)
    plot_system(Ĥ, split_states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)

abs2(split_states[:, end] ⋅ Ψ_tgt)

angle(split_states[:, end] ⋅ Ψ_tgt) / π

plot(theta_grid ./ π, V̂_tgt.diag ./ MHz, label="V")
_offset = minimum(V̂_tgt.diag ./ MHz)
plot!(theta_grid ./ π, 50 .* abs2.(split_states[:, end]) .+ _offset, label="|Ψ(T)|²")
plot!(theta_grid ./ π, 50 .* abs2.(Ψ_tgt) .+ _offset, label="|Ψ_tgt|²")
plot!(; xlabel="θ/π", ylabel="Energy (MHz)", xlim=(0.62, 0.63))

# ## Free time evolution

Ψ_free = Boost' * split_states[:, end];

Ĥ_free = evaluate(Ĥ, tlist, length(tlist) - 1);

V̂_free = Ĥ_free.V;

plot(theta_grid ./ π, V̂_free.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_free.diag ./ MHz)
plot!(theta_grid ./ π, 50 .* abs2.(Ψ_free) .+ _offset, label="|Ψ|²")

θ₀ = evaluate(Ĥ.V.phi, tlist, length(tlist) - 1)

Ψ_free_cut, theta_free = cut_grid(Ψ_free, theta_grid; θmin=(θ₀ + 0.1π), θmax=(θ₀ + 0.15π));

Ĥ_free = evaluate(
    rotating_tai_hamiltonian(tlist=[0, 1.0], theta=theta_free, phi=[0.0,]),
    [0, 1.0],
    1
);

plot(theta_free ./ π, Ĥ_free.V.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_free.diag ./ MHz)
plot!(theta_free ./ π, 50 .* abs2.(Ψ_free_cut) .+ _offset, label="|Ψ|²")

# ## Full scheme propagation

function eval_scheme(t_split, t_free, Ω)
    Ψ0 = prop_squeme(; dir=1, Ω=Ω)
    Ψ1 = prop_squeme(; dir=-1, Ω=Ω)
    Ψup = (Ψ0 + Ψ1)
    abs2(Ψup ⋅ Ψup) / 16
end

# ## Optimization

using QuantumControl

objective = Objective(initial_state=Ψ₀, target_state=Ψ_tgt, generator=Ĥ)

ω = get_controls(objective.generator)[1];

problem = ControlProblem(;
    objectives=[objective],
    tlist,
    pulse_options=IdDict(ω => Dict(:lambda_a => 1.0, :update_shape => (t -> 1.0)))
);

# +
#optimize(problem; prop_method=:cheby, method=:krotov, specrange_method=:arnoldi, J_T=QuantumControl.Functionals.J_T_sm)
