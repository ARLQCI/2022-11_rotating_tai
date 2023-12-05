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
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Map adiabaticity

# ## Hamiltonian

using QuantumPropagators
using QuantumPropagators.Storage: init_storage
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter

using Revise

using Plots

const 𝕚 = 1im;
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
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;

includet("./include/rotating_tai.jl")

includet("./include/split_propagator.jl")

function rotating_tai_hamiltonian(;
    tlist,
    θ,
    ω,  # function of time
    V₀,
    Ω=0.0,
    direction=1,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)

    V = Diagonal(V₀ .* cos.(m .* θ))

    dθ                                   = θ[2] - θ[1]
    nθ                                   = length(θ)
    pgrid::Vector{Float64}               = 2π * fftfreq(nθ, 1 / dθ)
    P::Diagonal{Float64,Vector{Float64}} = Diagonal(pgrid)
    K::Diagonal{Float64,Vector{Float64}} = Diagonal(pgrid .^ 2 / (2 * mass))

    _Ψ = Array{ComplexF64}(undef, nθ)
    fft_op = plan_fft!(_Ψ)
    ifft_op = plan_ifft!(_Ψ)
    transforms = (Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)

    K′::Diagonal{Float64,Vector{Float64}} = K - Ω * P

    if ω isa Number
        if direction == 1
            H = SplitGenerator(K′ + ω * P, V, transforms...)
        elseif direction == -1
            H = SplitGenerator(K′ - ω * P, V, transforms...)
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = SplitGenerator(hamiltonian(K′, (P, ω)), V, transforms...)
        elseif direction == -1
            H = SplitGenerator(hamiltonian(K′, (-P, ω)), V, transforms...)
        else
            error("direction must be ±1")
        end
    end
end

omega_ramp_up(t; w0=OMEGA_TARGET, t_r) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r) = w0 * cos(π * t / (2t_r))^2;

using QuantumPropagators.Controls: discretize_on_midpoints

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

choose_timesteps(100ms)

function propagate_splitting(
    separation_time,
    potential_depth;
    ret=:fidelity,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    theta_max=0.25π,
    theta_steps=1024,
    kwargs...
)
    nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, separation_time, length=nt))
    ω_func(t) = omega_ramp_up(t; w0=OMEGA_TARGET, t_r=separation_time)
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    Ĥ = rotating_tai_hamiltonian(
        tlist=tlist,
        V₀=potential_depth,
        θ=θ,
        ω=discretize_on_midpoints(ω_func, tlist)
    )
    if ret == :system
        return Ĥ, tlist
    end
    Ĥ₀ = evaluate(Ĥ, tlist, 1)
    Ψ₀ = get_ground_state(Ĥ₀, θ, π / 8, d=0.05, steps=10_000)
    if ret == :initial_state
        return Ψ₀, θ
    end
    Ĥ_tgt = evaluate(Ĥ, tlist, nt - 1)
    Ψ_tgt = get_ground_state(Ĥ_tgt, θ, π / 8, d=0.05, steps=10_000)
    if ret == :target
        return Ψ_tgt, θ
    end
    Ψ = propagate(Ψ₀, Ĥ, tlist; method=:splitprop, kwargs...)
    if ret == :propagation
        return Ψ
    end
    F = abs2(Ψ ⋅ Ψ_tgt)
    if ret == :fidelity
        return F
    else
        error("Invalid ret=$ret")
    end
end

# ## Sanity check

# ### Long separation time, deep potential

t_r = 100ms;
V0 = 2.2MHz;
scale = 300;

Ĥ, tlist = propagate_splitting(t_r, V0; ret=:system)
Ψ₀, θ = propagate_splitting(t_r, V0; ret=:initial_state)
V₀ = evaluate(Ĥ, tlist, 1).V.diag
plot(θ ./ π, V₀)
plot!(θ ./ π, scale .* real.(Ψ₀) .+ minimum(V₀))
plot!(θ ./ π, scale .* imag.(Ψ₀) .+ minimum(V₀))

Ψ_tgt, θ = propagate_splitting(t_r, V0; ret=:target)
Ψ_T = propagate_splitting(t_r, V0; ret=:propagation)
V_tgt = evaluate(Ĥ, tlist, 1000).V.diag
scale = 300
plot(θ ./ π, V_tgt, label="V(θ)")
#plot!(θ ./ π, scale .* real.(Ψ_tgt) .+ minimum(V_tgt), label="Re[Ψ_tgt]")
#plot!(θ ./ π, scale .* imag.(Ψ_tgt) .+ minimum(V_tgt), label="Im[Ψ_tgt]")
plot!(θ ./ π, scale .* abs2.(Ψ_tgt) .+ minimum(V_tgt), label="|Ψ_tgt|²")
#plot!(θ ./ π, scale .* real.(Ψ_T) .+ minimum(V_tgt), label="Re[Ψ(T)]")
#plot!(θ ./ π, scale .* imag.(Ψ_T) .+ minimum(V_tgt), label="Im[Ψ(T)]")
plot!(θ ./ π, scale .* abs2.(Ψ_T) .+ minimum(V_tgt), label="|Ψ(T)|²")

@time propagate_splitting(t_r, V0)

function psi_to_momentum(Ψ, θ)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    Ψ_p = fftshift(fft(Ψ))
    return Ψ_p, p
end

Ψ0_p, p = psi_to_momentum(Ψ₀, θ)
plot(p, abs2.(Ψ0_p), label="Ψ₀")
Ψ_p, p = psi_to_momentum(Ψ_tgt, θ)
plot!(p, abs2.(Ψ_p), label="target")
Ψ_T_p, p = psi_to_momentum(Ψ_T, θ)
plot!(p, abs2.(Ψ_T_p), label="Ψ(T)")
plot!(; xlabel="momentum", ylabel="amplitude", xlim=(-500, 500))
vline!([-EFFECTIVE_MASS * 10π / sec], color="black", label=raw"$Mω_0$")

EFFECTIVE_MASS * 10π / sec

# ### Short separation time, deep potential

t_r = 0.1μs;
V0 = 2.2MHz;
scale = 300;

Ĥ, tlist = propagate_splitting(t_r, V0; ret=:system)
Ψ₀, θ = propagate_splitting(t_r, V0; ret=:initial_state)
V₀ = evaluate(Ĥ, tlist, 1).V.diag
plot(θ ./ π, V₀)
plot!(θ ./ π, scale .* real.(Ψ₀) .+ minimum(V₀))
plot!(θ ./ π, scale .* imag.(Ψ₀) .+ minimum(V₀))

Ψ_tgt, θ = propagate_splitting(t_r, V0; ret=:target)
Ψ_T = propagate_splitting(t_r, V0; ret=:propagation)
V_tgt = evaluate(Ĥ, tlist, 1000).V.diag
scale = 300
plot(θ ./ π, V_tgt, label="V(θ)")
#plot!(θ ./ π, scale .* real.(Ψ_tgt) .+ minimum(V_tgt), label="Re[Ψ_tgt]")
#plot!(θ ./ π, scale .* imag.(Ψ_tgt) .+ minimum(V_tgt), label="Im[Ψ_tgt]")
plot!(θ ./ π, scale .* abs2.(Ψ_tgt) .+ minimum(V_tgt), label="|Ψ_tgt|²")
#plot!(θ ./ π, scale .* real.(Ψ_T) .+ minimum(V_tgt), label="Re[Ψ(T)]")
#plot!(θ ./ π, scale .* imag.(Ψ_T) .+ minimum(V_tgt), label="Im[Ψ(T)]")
plot!(θ ./ π, scale .* abs2.(Ψ_T) .+ minimum(V_tgt), label="|Ψ(T)|²")

Ψ0_p, p = psi_to_momentum(Ψ₀, θ)
plot(p, abs2.(Ψ0_p), label="Ψ₀")
Ψ_p, p = psi_to_momentum(Ψ_tgt, θ)
plot!(p, abs2.(Ψ_p), label="target")
Ψ_T_p, p = psi_to_momentum(Ψ_T, θ)
plot!(p, abs2.(Ψ_T_p); label="Ψ(T)", ls=:dash)
plot!(; xlabel="momentum", ylabel="amplitude", xlim=(-500, 500))

@time propagate_splitting(t_r, V0)

abs2(Ψ_T ⋅ Ψ_tgt)

propagate_splitting(t_r, V0; minimum_timesteps=100_000)

abs2(Ψ_T ⋅ Ψ₀)

# ### Long separation time, shallow potential

t_r = 100ms;
V0 = 0.1MHz;
scale = 1;

Ĥ, tlist = propagate_splitting(t_r, V0; ret=:system)
Ψ₀, θ = propagate_splitting(t_r, V0; ret=:initial_state)
V₀ = evaluate(Ĥ, tlist, 1).V.diag
plot(θ ./ π, V₀)
plot!(θ ./ π, scale .* real.(Ψ₀) .+ minimum(V₀))
plot!(θ ./ π, scale .* imag.(Ψ₀) .+ minimum(V₀))

Ψ_tgt, θ = propagate_splitting(t_r, V0; ret=:target)
Ψ_T = propagate_splitting(t_r, V0; ret=:propagation)
V_tgt = evaluate(Ĥ, tlist, 1000).V.diag
scale = 300
plot(θ ./ π, V_tgt, label="V(θ)")
#plot!(θ ./ π, scale .* real.(Ψ_tgt) .+ minimum(V_tgt), label="Re[Ψ_tgt]")
#plot!(θ ./ π, scale .* imag.(Ψ_tgt) .+ minimum(V_tgt), label="Im[Ψ_tgt]")
plot!(θ ./ π, scale .* abs2.(Ψ_tgt) .+ minimum(V_tgt), label="|Ψ_tgt|²")
#plot!(θ ./ π, scale .* real.(Ψ_T) .+ minimum(V_tgt), label="Re[Ψ(T)]")
#plot!(θ ./ π, scale .* imag.(Ψ_T) .+ minimum(V_tgt), label="Im[Ψ(T)]")
plot!(θ ./ π, scale .* abs2.(Ψ_T) .+ minimum(V_tgt), label="|Ψ(T)|²")

Ψ0_p, p = psi_to_momentum(Ψ₀, θ)
plot(p, abs2.(Ψ0_p), label="Ψ₀")
Ψ_p, p = psi_to_momentum(Ψ_tgt, θ)
plot!(p, abs2.(Ψ_p), label="target")
Ψ_T_p, p = psi_to_momentum(Ψ_T, θ)
plot!(p, abs2.(Ψ_T_p), label="Ψ(T)")
plot!(; xlabel="momentum", ylabel="amplitude", xlim=(-500, 500))

@time propagate_splitting(t_r, V0)

# ### Short separation time, shallow potential

t_r = 0.1μs;
V0 = 0.1MHz;
scale = 300;

Ĥ, tlist = propagate_splitting(t_r, V0; ret=:system)
Ψ₀, θ = propagate_splitting(t_r, V0; ret=:initial_state)
V₀ = evaluate(Ĥ, tlist, 1).V.diag
plot(θ ./ π, V₀)
plot!(θ ./ π, scale .* real.(Ψ₀) .+ minimum(V₀))
plot!(θ ./ π, scale .* imag.(Ψ₀) .+ minimum(V₀))

Ψ_tgt, θ = propagate_splitting(t_r, V0; ret=:target)
Ψ_T = propagate_splitting(t_r, V0; ret=:propagation)
V_tgt = evaluate(Ĥ, tlist, 1000).V.diag
scale = 300
plot(θ ./ π, V_tgt, label="V(θ)")
#plot!(θ ./ π, scale .* real.(Ψ_tgt) .+ minimum(V_tgt), label="Re[Ψ_tgt]")
#plot!(θ ./ π, scale .* imag.(Ψ_tgt) .+ minimum(V_tgt), label="Im[Ψ_tgt]")
plot!(θ ./ π, scale .* abs2.(Ψ_tgt) .+ minimum(V_tgt), label="|Ψ_tgt|²")
#plot!(θ ./ π, scale .* real.(Ψ_T) .+ minimum(V_tgt), label="Re[Ψ(T)]")
#plot!(θ ./ π, scale .* imag.(Ψ_T) .+ minimum(V_tgt), label="Im[Ψ(T)]")
plot!(θ ./ π, scale .* abs2.(Ψ_T) .+ minimum(V_tgt), label="|Ψ(T)|²")

Ψ0_p, p = psi_to_momentum(Ψ₀, θ)
plot(p, abs2.(Ψ0_p), label="Ψ₀")
Ψ_p, p = psi_to_momentum(Ψ_tgt, θ)
plot!(p, abs2.(Ψ_p), label="target")
Ψ_T_p, p = psi_to_momentum(Ψ_T, θ)
plot!(p, abs2.(Ψ_T_p); label="Ψ(T)", ls=:dash)
plot!(; xlabel="momentum", ylabel="amplitude", xlim=(-500, 500))

@time propagate_splitting(t_r, V0)

propagate_splitting(t_r, V0, theta_steps=2048, minimum_timesteps=100_000)

# #### Check Hamiltonian (DEBUG)

function plot_hamiltonian(H, tlist, n; component=:T, compare_to_n=nothing)
    Ĥ = evaluate(H, tlist, n)
    Y = getproperty(Ĥ, component).diag
    label = "$component (n=$n)"
    if !isnothing(compare_to_n)
        Y0 = getproperty(evaluate(H, tlist, compare_to_n), component).diag
        @. Y = Y - Y0
        label = "Δ$component (n=$compare_to_n → n=$n)"
    end
    plot(Y; label)
end

plot_hamiltonian(Ĥ, tlist, 1)

plot_hamiltonian(Ĥ, tlist, length(tlist) - 1)

plot_hamiltonian(Ĥ, tlist, length(tlist) - 1; compare_to_n=1)

# ## Map

# ### V0 = 0.1 - 2.2 MHz; sep time = 10⁻¹ - 10⁵ μs

potential_depth_values = collect(range(0.1MHz, 2.2MHz, length=106))
potential_depth_values ./ MHz

separation_time_orders_of_magnitude = collect(range(-1, 5, length=121))

separation_time_values = [10^x * μs for x in separation_time_orders_of_magnitude]

Threads.nthreads()

using ProgressMeter

function map_fidelity(potential_depth_values, separation_time_values; kwargs...)
    N = length(potential_depth_values)
    M = length(separation_time_values)
    F = zeros(N, M)
    progress = Progress(N * M)
    Threads.@threads for j = 1:M
        for i = 1:N
            t_r = separation_time_values[j]
            V0 = potential_depth_values[i]
            F[i, j] = propagate_splitting(t_r, V0; kwargs...)
            next!(progress)
        end
    end
    return F
end

include("./include/workflow.jl")

F = run_or_load("./data/2023-01-05_map_splitting_fidelity.npz"; force=false) do
    map_fidelity(potential_depth_values, separation_time_values)
end

# #### Countour Plot

contourf(
    separation_time_values ./ sec,
    potential_depth_values ./ MHz,
    F,
    tick_direction=:out,
    xminorticks=9,
    yminorticks=2,
    xaxis=:log10,
    ylabel="V₀ (MHz)",
    xlabel="separation time (seconds)",
    title=raw"Separation Fidelity $|⟨Ψ(t_r) | Ψ_{\textrm{tgt}}⟩|^2$",
)

# #### 3D Plot

# + active=""
# plotlyjs()

# + active=""
# surface(
#     separation_time_values ./ μs,
#     potential_depth_values ./ MHz,
#     F,
#     #tick_direction=:out,
#     #xminorticks=9,
#     #yminorticks=2,
#     xaxis=:log10,
#     ylabel="V₀ (MHz)",
#     xlabel="separation time (μs)",
#     title=raw"Separation Fidelity $|⟨Ψ(t_r) | Ψ_{\textrm{tgt}}⟩|^2$",
#     size=(1000, 600),
# )
# -

# ### Lower values of V0

potential_depth_values_low = collect(range(0MHz, 0.1MHz, step=0.01MHz))[1:end-1]
potential_depth_values_low ./ MHz

F_lowV0 = run_or_load("./data/2023-01-05_map_splitting_fidelity_lowV0.npz"; force=false) do
    map_fidelity(potential_depth_values_low, separation_time_values)
end

F_lowV0

potential_depth_values_combined = [potential_depth_values_low..., potential_depth_values...]
potential_depth_values_combined ./ MHz

F_combined = vcat(F_lowV0, F)

clamp.(F_combined, 0.5, 1.0)

contourf(
    separation_time_values ./ sec,
    potential_depth_values_combined ./ MHz,
    #F_combined,
    clamp.(F_combined, 0.5, 1.0),
    tick_direction=:out,
    xminorticks=9,
    yminorticks=2,
    xaxis=:log10,
    ylabel="V₀ (MHz)",
    xlabel="separation time (seconds)",
    title=raw"Separation Fidelity $|⟨Ψ(t_r) | Ψ_{\textrm{tgt}}⟩|^2$",
)

function p_matrix(theta_grid::Vector{Float64})
    nx::Int64 = length(theta_grid)
    dx::Float64 = theta_grid[2] - theta_grid[1]
    p̂ = zeros(ComplexF64, nx, nx)
    @inbounds for i = 1:nx
        for j = 1:nx
            if i ≠ j
                p̂[j, i] = ((-1)^(j - i)) / sin(π * (j - i) / nx)
            end
        end
    end
    lmul!(-1im * π / (nx * dx), p̂)
    return Hermitian(p̂)
end

function rotating_tai_hamiltonian_coord(;
    tlist,
    θ,
    ω,
    Ω=0.0,
    direction=1,
    V₀,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)

    V = Diagonal(V₀ .* cos.(m .* θ))

    dθ = θ[2] - θ[1]
    nθ = length(θ)
    P = p_matrix(θ)
    K = P^2 / (2 * mass)

    K′ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = K′ + ω * P + V
        elseif direction == -1
            H = K′ - ω * P + V
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = hamiltonian(K′ + V, (P, ω))
        elseif direction == -1
            H = hamiltonian(K′ + V, (-P, ω))
        else
            error("direction must be ±1")
        end
    end
    return H
end

function get_ground_state(; θ, V₀, ω=0.0, Ω=0.0)
    H₀ = rotating_tai_hamiltonian_coord(; tlist=[0.0], θ, ω, Ω, V₀)
    eigensys = eigen(H₀)
    Ψ₀ = eigensys.vectors[:, 1]
    return Ψ₀
end

function free_time_evolution(Ψ::Vector{ComplexF64}, H::AbstractMatrix, Δt::Float64)
    eigensys = eigen(H)
    U = eigensys.vectors
    λ = eigensys.values
    Ψ_out = U * (exp.(-1im .* λ .* Δt) .* (U' * Ψ))
    return Ψ_out
end

"""Propagate Ψ₀ defined on a single surface.

Ψ₀ may or may not be normalized. In general, the population on the surface,
‖Ψ₀‖², is preserved.
"""
function prop_scheme(;
    Ψ₀,
    ω₀,
    θ,
    t_r,
    t_loop,
    V0,
    dir=1,
    Ω=0.0,
    m=N_SITES,
    mass=EFFECTIVE_MASS,
    method=:splitprop,
    specrange_method=:arnoldi,
    nt_loop=2,
    ret=:state,
    storages=nothing,
    kwargs...
)
    nt = choose_timesteps(t_r)
    tlist_ramp_up = collect(range(0, t_r, length=nt))
    Ĥ_ramp_up = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_up,
        θ,
        Ω,
        V₀=V0,
        ω=discretize_on_midpoints(t -> omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up),
        direction=dir
    )

    if nt_loop ≤ 2
        tlist_loop = [t_r, t_r + t_loop]
        Ĥ_loop = rotating_tai_hamiltonian_coord(;
            tlist=tlist_loop,
            θ,
            Ω,
            ω=ω₀,
            V₀=V0,
            direction=dir
        )
    else
        tlist_loop = collect(range(t_r, t_r + t_loop, length=nt_loop))
        Ĥ_loop =
            rotating_tai_hamiltonian(; tlist=tlist_loop, θ, Ω, ω=ω₀, V₀=V0, direction=dir)
    end

    tlist_ramp_down =
        collect(range(t_r + t_loop, 2 * t_r + t_loop, length=length(tlist_ramp_up)))
    Ĥ_ramp_down = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_down,
        θ,
        Ω,
        V₀=V0,
        ω=discretize_on_midpoints(
            t -> omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r),
            tlist_ramp_down
        ),
        direction=dir
    )
    if ret == :tlist
        return (tlist_ramp_up, tlist_loop, tlist_ramp_down)
    end

    if isnothing(storages)
        storages = (nothing, nothing, nothing)
    else
        if nt_loop ≤ 2
            error("storages can only be used for nt_loop > 2")
        end
    end

    Ψ = propagate(
        Ψ₀,
        Ĥ_ramp_up,
        tlist_ramp_up;
        method,
        specrange_method,
        storage=storages[1],
        kwargs...
    )
    if nt_loop ≤ 2
        Ψ = free_time_evolution(Ψ, Ĥ_loop, t_loop)
    else
        Ψ = propagate(
            Ψ₀,
            Ĥ_loop,
            tlist_loop;
            method,
            specrange_method,
            storage=storages[2],
            kwargs...
        )
    end
    Ψ = propagate(
        Ψ,
        Ĥ_ramp_down,
        tlist_ramp_down;
        method,
        specrange_method,
        storage=storages[3],
        kwargs...
    )
    if ret == :state
        return Ψ
    else
        error("Invalid ret=$ret")
    end
end

surface_pop(Ψ) = norm(Ψ)^2

function eval_scheme(; ω₀, θ, t_r, t_loop, V0, Ω, parallel=true)

    U_πhalf = [
        1/√2  𝕚/√2
        𝕚/√2  1/√2
    ]

    Ψright = get_ground_state(; θ, Ω, V₀=V0)
    Ψleft = zeros(ComplexF64, length(Ψright))

    Ψright, Ψleft = U_πhalf * [Ψright, Ψleft]

    if parallel
        prop_right =
            Threads.@spawn prop_scheme(; Ψ₀=Ψright, ω₀, θ, t_r, t_loop, V0, dir=1, Ω=Ω)
        prop_left =
            Threads.@spawn prop_scheme(; Ψ₀=Ψleft, ω₀, θ, t_r, t_loop, V0, dir=-1, Ω=Ω)
        Ψright = fetch(prop_right)
        Ψleft = fetch(prop_left)
    else
        Ψright = prop_scheme(; Ψ₀=Ψright, ω₀, θ, t_r, t_loop, V0, dir=1, Ω=Ω)
        Ψleft = prop_scheme(; Ψ₀=Ψleft, ω₀, θ, t_r, t_loop, V0, dir=-1, Ω=Ω)
    end

    Ψleft, Ψright = U_πhalf * [Ψleft, Ψright]
    return surface_pop(Ψright)
end

function scan_Ω(Ω_list; ω₀, θ, t_r, t_loop, V0, parallel=true, show_progress=true)
    P_list = Float64[]
    if show_progress
        progress = Progress(length(Ω_list))
    end
    for Ω in Ω_list
        P = eval_scheme(; ω₀, θ, t_r, t_loop, V0, Ω, parallel)
        push!(P_list, P)
        show_progress && next!(progress)

    end
    return P_list
end

# ### Normal wait time

Ω_list = collect(range(0, 0.1 / sec, length=41))
P_list = run_or_load("./data/2023-01-05_map_splitting_scan.npz"; force=false) do
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    scan_Ω(Ω_list; ω₀=OMEGA_TARGET, θ, t_r=0.1μs, V0=0.1MHz, t_loop=900ms)
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false, ylim=(0, 1))

function contrast(populations)
    P_max = maximum(populations)
    P_min = minimum(populations)
    return (P_max - P_min) / (P_max + P_min)
end

println("Contrast = $(contrast(P_list))")

# ### Long wait time

Ω_list = collect(range(0, 0.1 / sec, length=41))
P_list = run_or_load("./data/2023-01-05_map_splitting_scan_long_loop.npz"; force=false) do
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    scan_Ω(Ω_list; ω₀=OMEGA_TARGET, θ, t_r=0.1μs, V0=0.1MHz, t_loop=1800ms)
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false, ylim=(0, 1))

println("Contrast = $(contrast(P_list))")

# ### Contrast depending on wait time

function scan_contrast(loop_times, Ω_list)
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    C = zeros(length(loop_times))
    progress = Progress(length(loop_times))
    Threads.@threads for i = 1:length(loop_times)
        t_loop = loop_times[i]
        P_list = scan_Ω(
            Ω_list;
            ω₀=OMEGA_TARGET,
            θ,
            t_r=0.1μs,
            V0=0.1MHz,
            t_loop=float(t_loop),
            parallel=false,
            show_progress=false
        )
        C[i] = contrast(P_list)
        next!(progress)
    end
    return C
end

loop_times = collect(range(400ms, 3000ms, step=100ms))
contrast_list =
    run_or_load("./data/2023-01-05_map_splitting_scan_contrast.npz"; force=false) do
        scan_contrast(loop_times, Ω_list)
    end

plot(
    loop_times ./ ms,
    contrast_list,
    xlabel="loop time (ms)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz",
)

loop_times = collect(range(400ms, 3000ms, step=10ms))
contrast_list =
    run_or_load("./data/2023-01-05_map_splitting_scan_contrast_highrez.npz"; force=false) do
        scan_contrast(loop_times, Ω_list)
    end
plot(
    loop_times ./ ms,
    contrast_list,
    xlabel="loop time (ms)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz",
)


loop_times = collect(range(400ms, 3000ms, step=1ms))
contrast_list = run_or_load(
    "./data/2023-01-05_map_splitting_scan_contrast_highrez2.npz";
    force=false
) do
    scan_contrast(loop_times, Ω_list)
end

plot(
    loop_times ./ ms,
    contrast_list,
    xlabel="loop time (ms)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz",
)

loop_times = collect(range(850ms, 950ms, step=0.1ms))
@show length(loop_times)
contrast_list =
    run_or_load("./data/2023-01-05_map_splitting_scan_contrast_850_950.npz"; force=false) do
        scan_contrast(loop_times, Ω_list)
    end

plot(
    loop_times ./ ms,
    contrast_list,
    xlabel="loop time (ms)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz",
)

loop_times = collect(range(895ms, 905ms, step=0.01ms))
@show length(loop_times)
contrast_list =
    run_or_load("./data/2023-01-05_map_splitting_scan_contrast_895_905.npz"; force=false) do
        scan_contrast(loop_times, Ω_list)
    end

plot(
    loop_times ./ ms,
    contrast_list,
    xlabel="loop time (ms)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz",
)

# ### Contrast for exact cycles

function angular_displacement(; ω₀=OMEGA_TARGET, t_r=0.1μs, t_loop)

    nt = choose_timesteps(t_r)
    tlist_ramp_up = collect(range(0, t_r, length=nt))
    dt_up = tlist_ramp_up[2] - tlist_ramp_up[1]
    ω_up = discretize_on_midpoints(t -> omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up)

    tlist_ramp_down = collect(range(t_r + t_loop, 2 * t_r + t_loop, length=nt))
    dt_down = tlist_ramp_down[2] - tlist_ramp_down[1]
    ω_down = discretize_on_midpoints(
        t -> omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r),
        tlist_ramp_down
    )

    Φ_up = sum(ω_up) * dt_up
    Φ_loop = t_loop * ω₀
    Φ_down = sum(ω_down) * dt_down
    Φ = Φ_up + Φ_loop + Φ_down

    Φtgt = round(Φ / π) * π
    t_loop_opt = (Φtgt - (Φ_up + Φ_down)) / ω₀
    if abs(t_loop - t_loop_opt) > 1e-8
        @warn("angular displacement not aligned. Change t_loop to $(t_loop_opt/ms)ms")
    end

    return Φ
end

# +
function t_loop_for_cycle(n_cycles; ω₀=OMEGA_TARGET, t_r=0.1μs)

    nt = choose_timesteps(t_r)
    tlist_ramp_up = collect(range(0, t_r, length=nt))
    dt_up = tlist_ramp_up[2] - tlist_ramp_up[1]
    ω_up = discretize_on_midpoints(t -> omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up)

    t_loop = 0.0
    tlist_ramp_down = collect(range(t_r + t_loop, 2 * t_r + t_loop, length=nt))
    dt_down = tlist_ramp_down[2] - tlist_ramp_down[1]
    ω_down = discretize_on_midpoints(
        t -> omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r),
        tlist_ramp_down
    )

    Φ_up = sum(ω_up) * dt_up
    Φ_down = sum(ω_down) * dt_down

    Φtgt = n_cycles * π
    t_loop = (Φtgt - (Φ_up + Φ_down)) / ω₀
    return t_loop

end
# -

angular_displacement(; t_loop=t_loop_for_cycle(10)) / π

t_loop_for_cycle(1) / ms

t_loop_for_cycle(1) / sec

t_loop_for_cycle(2) / sec

t_loop_for_cycle(100) / sec

# #### 1 Cycle

n_cycles = 1
Ω_list = collect(range(0, (0.5 / n_cycles) / sec, length=21))
P_list = run_or_load("./data/2023-01-05_map_splitting_scan_1_loop.npz"; force=false) do
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    scan_Ω(
        Ω_list;
        ω₀=OMEGA_TARGET,
        θ,
        t_r=0.1μs,
        V0=0.1MHz,
        t_loop=t_loop_for_cycle(n_cycles)
    )
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false, ylim=(0, 1))

contrast(P_list)

# #### 10 Cycles

n_cycles = 10
Ω_list = collect(range(0, (0.5 / n_cycles) / sec, length=41))
P_list = run_or_load("./data/2023-01-05_map_splitting_scan_10_loop.npz"; force=false) do
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    scan_Ω(
        Ω_list;
        ω₀=OMEGA_TARGET,
        θ,
        t_r=0.1μs,
        V0=0.1MHz,
        t_loop=t_loop_for_cycle(n_cycles)
    )
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false, ylim=(0, 1))

# #### 100 Cycles

n_cycles = 100
Ω_list = collect(range(0, (0.5 / n_cycles) / sec, length=21))
P_list = run_or_load("./data/2023-01-05_map_splitting_scan_100_loop.npz"; force=false) do
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    scan_Ω(
        Ω_list;
        ω₀=OMEGA_TARGET,
        θ,
        t_r=0.1μs,
        V0=0.1MHz,
        t_loop=t_loop_for_cycle(n_cycles)
    )
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false, ylim=(0, 1))

contrast(P_list)

# #### Scan

function scan_contrast_for_cycles(cycle_numbers)
    theta_max = 0.25π
    theta_steps = 1024
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    C = zeros(length(cycle_numbers))
    progress = Progress(length(cycle_numbers))
    Threads.@threads for i = 1:length(cycle_numbers)
        n_cycles = cycle_numbers[i]
        Ω_list = collect(range(0, (0.5 / n_cycles) / sec, length=21))
        t_loop = t_loop_for_cycle(n_cycles)
        P_list = scan_Ω(
            Ω_list;
            ω₀=OMEGA_TARGET,
            θ,
            t_r=0.1μs,
            V0=0.1MHz,
            t_loop=float(t_loop),
            parallel=false,
            show_progress=false
        )
        C[i] = contrast(P_list)
        next!(progress)
    end
    return C
end

# ##### Every 5

cycle_numbers = [1, range(5, 500, step=5)...]

contrast_list =
    run_or_load("./data/2023-01-05_map_splitting_scan_contrast_cycles.npz"; force=false) do
        scan_contrast_for_cycles(cycle_numbers)
    end

plot(
    cycle_numbers,
    contrast_list;
    marker="o",
    xlabel="cycles (≈ 0.1 seconds)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz"
)

# ##### Every 1

cycle_numbers = collect(range(1, 100, step=1))

contrast_list = run_or_load(
    "./data/2023-01-05_map_splitting_scan_contrast_cycles_1_100.npz";
    force=false
) do
    scan_contrast_for_cycles(cycle_numbers)
end

plot(
    cycle_numbers,
    contrast_list;
    marker="o",
    xlabel="cycles (≈ 0.1 seconds)",
    ylabel="contrast",
    legend=false,
    title=raw"Contrast for $t_r$ = 0.1 μs, V₀ = 100 kHz"
)

# ## Full scheme dynamics

# +
function propagate_full_dynamics(;
    ω₀=OMEGA_TARGET,
    θ=collect(range(0, 0.25π, length=1024)),
    t_r=0.1μs,
    t_loop=t_loop_for_cycle(10),
    V0=0.1MHz,
    Ω=0.0,
    nt_loop=1001,
    method=:cheby,
    parallel=true,
    uniform_dt_tolerance=1e-8
)

    U_πhalf = [
        1/√2  𝕚/√2
        𝕚/√2  1/√2
    ]

    Ψright = get_ground_state(; θ, Ω, V₀=V0)
    Ψleft = zeros(ComplexF64, length(Ψright))

    Ψright, Ψleft = U_πhalf * [Ψright, Ψleft]

    time_grids = prop_scheme(;
        Ψ₀=Ψright,
        ω₀,
        θ,
        t_r,
        t_loop,
        V0,
        dir=1,
        Ω=Ω,
        method,
        nt_loop,
        ret=:tlist
    )
    tlist_ramp_up, tlist_loop, tlist_ramp_down = time_grids
    # return tlist_ramp_up, tlist_loop, tlist_ramp_down # DEBUG

    storages_right = (
        init_storage(Ψright, tlist_ramp_up),
        init_storage(Ψright, tlist_loop),
        init_storage(Ψright, tlist_ramp_down)
    )
    storages_left = (
        init_storage(Ψright, tlist_ramp_up),  # Ψright is fine for initialization
        init_storage(Ψright, tlist_loop),
        init_storage(Ψright, tlist_ramp_down)
    )

    if parallel
        prop_right = Threads.@spawn prop_scheme(;
            Ψ₀=Ψright,
            ω₀,
            θ,
            t_r,
            t_loop,
            V0,
            dir=1,
            Ω=Ω,
            method,
            nt_loop,
            storages=storages_right,
            uniform_dt_tolerance
        )
        prop_left = Threads.@spawn prop_scheme(;
            Ψ₀=Ψleft,
            ω₀,
            θ,
            t_r,
            t_loop,
            V0,
            dir=-1,
            Ω=Ω,
            method,
            nt_loop,
            storages=storages_left,
            uniform_dt_tolerance
        )
        Ψright = fetch(prop_right)
        Ψleft = fetch(prop_left)
    else
        Ψright = prop_scheme(;
            Ψ₀=Ψright,
            ω₀,
            θ,
            t_r,
            t_loop,
            V0,
            dir=1,
            Ω=Ω,
            method,
            nt_loop,
            storages=storages_right,
            uniform_dt_tolerance
        )
        Ψleft = prop_scheme(;
            Ψ₀=Ψleft,
            ω₀,
            θ,
            t_r,
            t_loop,
            V0,
            dir=-1,
            Ω=Ω,
            method,
            nt_loop,
            storages=storages_left,
            uniform_dt_tolerance
        )
    end

    return time_grids, storages_right, storages_left

end
# -

using FileIO

full_dynamics = run_or_load(
    "./data/2023-01-05_map_splitting_full_dynamics.jld2";
    save=FileIO.save,
    load=FileIO.load,
    force=false
) do
    time_grids, storages_right, storages_left = propagate_full_dynamics()
    return Dict(
        "time_grids" => time_grids,
        "storages_right" => storages_right,
        "storages_left" => storages_left,
    )
end

sum([length(tlist) for tlist in full_dynamics["time_grids"]])

V₀

θ

using Printf

function plot_full_dynamics_frame(full_dynamics, n; V₀=V₀, θ=θ, scale=10)
    tlist_ramp_up, tlist_loop, tlist_ramp_down = full_dynamics["time_grids"]
    seg_labels = ["split", "loop", "recomb"]
    segment = 1
    if n > length(tlist_ramp_up)
        segment = 2
        n = n - length(tlist_ramp_up)
    end
    if n > length(tlist_loop)
        segment = 3
        n = n - length(tlist_loop)
    end
    t = full_dynamics["time_grids"][segment][n] / sec
    offset = minimum(V₀) / MHz
    Ψ_left = full_dynamics["storages_left"][segment]
    Ψ_right = full_dynamics["storages_right"][segment]
    fig = plot(
        θ ./ π,
        V₀ / MHz,
        xlabel="θ/π",
        ylabel="Energy (MHz)",
        label="",
        title="[$(seg_labels[segment])] t=$(@sprintf("%.2e", t)) s"
    )
    plot!(fig, θ ./ π, scale * abs2.(Ψ_left[:, n]) .+ offset, label="|Ψₗ|²")
    plot!(fig, θ ./ π, scale * abs2.(Ψ_right[:, n]) .+ offset, label="|Ψᵣ|²")
    plot!(fig, ylim=(-0.11, 0.18), xlim=(0.10, 0.15))
end

nt_total = sum([length(tlist) for tlist in full_dynamics["time_grids"]])
anim = @animate for n = 1:nt_total
    plot_full_dynamics_frame(full_dynamics, n)
end
gif(anim, "anim.gif", fps=10)

mp4(anim, "anim.mp4", fps=10)
