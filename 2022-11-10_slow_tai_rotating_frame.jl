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
#     display_name: Julia 1.8 (4 threads)
#     language: julia
#     name: julia-1.8-multithread
# ---

# # Rotating TAI with ω(t) - adiabatic splitting in the moving frame

# Here, we use $\omega(t)$ as the control field and $\phi(t) = \int \omega(t) dt$ as the control amplitude.
#
# We also use the Hamiltonian in the moving frame,
#
# \begin{equation}
# \tilde{H}_{\pm} (t)
# = -\frac{\hbar^2}{2mR^2}\frac{\partial^2}{\partial \theta^2} + V_0 \cos\left(m \theta\right) + i \hbar \omega_{\pm}(t) \frac{\partial}{\partial \theta}
# = \left((\hat{T} - \Omega \, \hat{p})  \mp \omega(t) \, \hat{p}\right) + \hat{V} 
# \end{equation}

# ## Hamiltonian

using QuantumPropagators
using LinearAlgebra
using FFTW
using ProgressMeter

import QuantumControl.Controls: discretize, discretize_on_midpoints, evalcontrols

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
const SEPARATION_TIME = 100ms;
const OMEGA_TARGET = 2π/sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;

includet("./rotating_tai.jl")


includet("./split_propagator.jl")

tlist = collect(range(0, SEPARATION_TIME, length=(Int(SEPARATION_TIME÷ms)*1000+1)));
theta_grid = collect(range(0, 0.25π, length=512));

length(tlist)

# **TODO: choose number of time grid points so that cheby is stable and split-prop gives the same result**

println("Storage for full Hamiltonian matrix: $(sizeof(zeros(Float64, length(theta_grid), length(theta_grid))) / 1024^2) MB")

function rotating_tai_hamiltonian(;
        tlist, θ, ω, Ω=0.0, direction=1,
        V₀=POTENTIAL_DEPTH, m=N_SITES, mass=EFFECTIVE_MASS
)

    V = Diagonal(V₀ .* cos.(m .* θ))

    dθ = θ[2] - θ[1]
    nθ = length(θ)
    pgrid = 2π * fftfreq(nθ, 1 / dθ)
    P = Diagonal(pgrid)
    K = Diagonal(pgrid .^ 2 / (2 * mass))

    _Ψ = Array{ComplexF64}(undef, nθ)
    fft_op = plan_fft!(_Ψ)
    ifft_op = plan_ifft!(_Ψ)
    transforms = (Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)

    K̃ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = SplitGenerator(K̃ + ω * P , V, transforms...)
        elseif direction == -1
            H = SplitGenerator(K̃ - ω * P , V, transforms...)
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = SplitGenerator(hamiltonian(K̃, (P, ω)), V, transforms...)
        elseif direction == -1
            H = SplitGenerator(hamiltonian(K̃, (-P, ω)), V, transforms...)
        else
            error("direction must be ±1")
        end
    end
end

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π*t/(2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π*t/(2t_r))^2;

function evaluate(generator, tlist, n)
    vals_dict = IdDict(c => c[n] for c ∈ getcontrols(generator))
    evalcontrols(generator, vals_dict, tlist, n)
end

plot(
    tlist ./ sec,
    discretize(omega_ramp_up.(tlist), tlist) / (2π/sec);
    legend=:topleft, label="ω(t)", xlabel="time (sec)",
    ylabel="angular velocity (2π/sec)"
)

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    θ=theta_grid,
    ω=discretize_on_midpoints(omega_ramp_up, tlist)
);

typeof(Ĥ)

typeof(Ĥ.T)

typeof(Ĥ.V)

Ĥ₀ = evaluate(Ĥ, tlist, 1)
V̂₀ = Ĥ₀.V;

# ## Calculate initial state

Ψ₀ = get_ground_state(Ĥ₀, theta_grid, π/8,  d=0.05, steps=10_000);

plot(theta_grid./π, V̂₀.diag./MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀.diag./MHz)
plot!(theta_grid./π, 50 .* abs2.(Ψ₀).+_offset, label="|Ψ₀|²", xlim=(0,0.3), legend=:right)

println("⟨θ⟩_Ψ₀ = $(round(abs(dot(Ψ₀, Diagonal(theta_grid), Ψ₀) / π), digits=10))π")

# ## Target state

Ĥ_tgt = evaluate(Ĥ, tlist, length(tlist)-1);

typeof(Ĥ.T)

typeof(Ĥ₀.T)

V̂_tgt = Ĥ_tgt.V;

Ψ_tgt = get_ground_state(Ĥ_tgt, theta_grid, π/8,  d=0.05, steps=10_000);

norm(Ψ_tgt - Ψ₀)

norm(abs2.(Ψ_tgt) - abs2.(Ψ₀))

plot(theta_grid./π, V̂_tgt.diag./MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_tgt.diag./MHz)
plot!(theta_grid./π, 50 .* abs2.(Ψ₀).+_offset, label="|Ψ₀|²")
plot!(theta_grid./π, 50 .* abs2.(Ψ_tgt).+_offset, label="|Ψ_tgt|²")
plot!(;xlim=(0.12, 0.13))

plot(theta_grid./π, V̂_tgt.diag./MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_tgt.diag./MHz)
plot!(theta_grid./π, 50 .* real.(Ψ₀).+_offset, label="Re[Ψ₀]")
plot!(theta_grid./π, 50 .* real.(Ψ_tgt).+_offset, label="Re[Ψ_tgt]")
plot!(;xlim=(0.12, 0.13))

plot(theta_grid./π, real.(Ψ₀) - real.(Ψ_tgt), marker=true)

# ## Propagation (Split Propagator)

split_states = propagate(Ψ₀, Ĥ, tlist; method=:splitprop, specrange_method=:arnoldi, storage=true, showprogress=true);

function plot_system(generator, states, theta_grid, tlist, n; psi_scale=50)
    t = tlist[n]
    Ĥ = generator
    V = evaluate(Ĥ, tlist, min(n, length(tlist)-1)).V.diag
    offset = minimum(V/MHz)
    Ψ = states[:,n]
    fig = plot(theta_grid./π, V/MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
    plot!(fig, theta_grid./π, psi_scale*abs2.(Ψ).+offset, label="|Ψ|²")
    plot!(title="t=$(t/ms)ms")
end

anim = @animate for n=1:(length(tlist) ÷ 100):length(tlist)
    plot_system(Ĥ, split_states, theta_grid, tlist, n)
end
gif(anim, "anim.gif", fps=10)

abs2(split_states[:,end] ⋅ Ψ_tgt)

angle(split_states[:,end] ⋅ Ψ_tgt) / π

plot(theta_grid./π, V̂_tgt.diag./MHz, label="V")
_offset = minimum(V̂_tgt.diag./MHz)
plot!(theta_grid./π, 50 .* abs2.(split_states[:,end]).+_offset, label="|Ψ(T)|²")
plot!(theta_grid./π, 50 .* abs2.(Ψ_tgt).+_offset, label="|Ψ_tgt|²")
plot!(; xlabel="θ/π", ylabel="Energy (MHz)")

# ## Free time evolution

Ψ_free = split_states[:,end];

Ĥ_free = evaluate(Ĥ, tlist, length(tlist)-1);

V̂_free = Ĥ_free.V;

plot(theta_grid./π, V̂_free.diag./MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_free.diag./MHz)
plot!(theta_grid./π, 50 .* abs2.(Ψ_free).+_offset, label="|Ψ|²")

split_states_free = propagate(
    Ψ_free, Ĥ_free,
    collect(range(0, 800ms, length=101));
    method=:cheby, specrange_method=:arnoldi,
    storage=true, showprogress=true
);

plot(abs2.(split_states_free[:,end]))

split_states_free[:,begin] ⋅ split_states_free[:,end]

# ## Full scheme propagation

function prop_scheme(;ω₀, θ, t_r, t_loop, dir=1, Ω=0.0, V0=POTENTIAL_DEPTH, m=N_SITES, mass=EFFECTIVE_MASS)
    @assert m == N_SITES
    specrange_method = :arnoldi
    tlist_ramp_up = collect(range(0, t_r, length=(Int(t_r÷ms)*1000+1)))
    Ĥ_ramp_up = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_up, θ, Ω,
        ω=discretize_on_midpoints(t->omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up)
    )
    #t_loop = ... # TODO
    Ĥ₀ = evaluate(Ĥ, tlist_ramp_up, 1)
    Ψ₀ = get_ground_state(Ĥ₀, θ, π/8,  d=0.05, steps=10_000);
    Ψ = propagate(Ψ₀, Ĥ_ramp_up, tlist_ramp_up; method=:splitprop, specrange_method, show_progress=true)
    
    tlist_loop = collect(range(t_r, t_r + t_loop, length=101))
    Ĥ_loop = rotating_tai_hamiltonian(;tlist=tlist_loop, θ, Ω, ω=(dir > 0 ? ω₀ : -ω₀))
    Ψ = propagate(Ψ, Ĥ_loop, tlist_loop; method=:cheby, specrange_method, show_progress=true)
    
    tlist_ramp_down = collect(range(t_r + t_loop, 2*t_r + t_loop, length=length(tlist_ramp_up)))
    Ĥ_ramp_down = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_down, θ, Ω, 
        ω=discretize_on_midpoints(t->omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r), tlist_ramp_down)
    )
    Ψ = propagate(Ψ, Ĥ_ramp_down, tlist_ramp_down; method=:splitprop, specrange_method, show_progress=true)
    return Ψ
end

function eval_scheme(;ω₀, θ, t_r, t_loop, Ω)
    Ψ0 = Threads.@spawn prop_scheme(;ω₀, θ, t_r, t_loop, dir=1, Ω=Ω)
    Ψ1 = Threads.@spawn prop_scheme(;ω₀, θ, t_r, t_loop, dir=-1, Ω=Ω)
    Ψup = (fetch(Ψ0) + fetch(Ψ1))
    abs2(Ψup ⋅ Ψup)/16
end

Ω_list = collect(range(0, 0.5/sec, length=41));

function scan_Ω(Ω_list)
    P_list = Float64[]
    @showprogress for Ω in Ω_list
        P = eval_scheme(;ω₀=OMEGA_TARGET, θ=theta_grid, t_r=SEPARATION_TIME, t_loop=800ms, Ω)
        push!(P_list, P)
    end
    return P_list
end

P_list = scan_Ω(Ω_list)

plot(Ω_list / (1/sec), P_list, xlabel="Ω (1/sec)", legend=false)
