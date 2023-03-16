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

# #  OCT for `t_r=150μs`, `V0=0.2MHz` with variable Potential

# We are optimizing for point (5) at 0.2 MHz in `2023-02-01_map_analysis.ipynb`, varying both ω(t), not V₀(t)

using QuantumPropagators
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
const SEPARATION_TIME = 150μs;
const LOOP_TIME = 900ms;
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 0.2MHz;
const MOMENTUM_TARGET = - EFFECTIVE_MASS * OMEGA_TARGET;

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

using QuantumPropagators.Controls: discretize, discretize_on_midpoints

# ## Hamiltonian

using QuantumPropagators.Controls: discretize, discretize_on_midpoints

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

function propagate_splitting(;
    separation_time=SEPARATION_TIME,
    evolution_time=0.0,
    potential_depth=POTENTIAL_DEPTH,
    omega_target=OMEGA_TARGET,
    number_of_sites=N_SITES,
    mass=EFFECTIVE_MASS,
    ret=:fidelity,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    theta_max=0.25π,
    theta_steps=1024,
    ω=nothing,  # optimized ω(t); if nothing: guess ramp to omega_target
    η=nothing,
    kwargs...
)
    T = separation_time + evolution_time
    nt = choose_timesteps(T; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, T, length=nt))


    function ω_ampl_guess(t)
        if t <= separation_time
            omega_ramp_up(t; w0=omega_target, t_r=separation_time)
        else
            omega_target
        end
    end

    if isnothing(ω)
        ω = ω_ampl_guess
    end

    if ret == :omega
        return ω
    end

    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    if !isnothing(η)
        η = discretize_on_midpoints(η, tlist)
    end
    Ĥ = rotating_tai_hamiltonian(;
        tlist,
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        scale_potential=η,
        ω=discretize_on_midpoints(ω, tlist)
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
    if ret == :H_tgt
        return Ĥ_tgt
    end
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

HAMILTONIAN, TIME_GRID = propagate_splitting(;
    η=(t -> 1.0), ret=:system
);

plot(get_controls(HAMILTONIAN)[1])

plot(get_controls(HAMILTONIAN)[2])

# ## Initial state

INITIAL_STATE, THETA_GRID = propagate_splitting(;
    η=(t -> 1.0), ret=:initial_state
);

plot(abs2.(INITIAL_STATE))

# ## Target state

TARGET_STATE, _ = propagate_splitting(;
    η=(t -> 1.0), ret=:target
);

function psi_to_momentum(Ψ, θ)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    Ψ_p = fftshift(fft(Ψ))
    return Ψ_p, p
end

function plot_states_in_momentum_space(
    θ,
    states...;
    labels=["Ψ$i" for i = 1:length(states)],
    marker=(-EFFECTIVE_MASS * OMEGA_TARGET),
    marker_label=raw"$Mω_0$",
    kwargs...
)
    fig = plot(; xlabel="momentum", ylabel="amplitude", kwargs...)
    for (i, Ψ) in enumerate(states)
        Ψ_momentum, momentum_grid = psi_to_momentum(Ψ, θ)
        plot!(fig, momentum_grid, abs2.(Ψ_momentum), label=labels[i])
    end
    if !isnothing(marker)
        vline!(fig, [marker], color="black", label=marker_label)
    end
    fig
end

plot_states_in_momentum_space(
    THETA_GRID,
    INITIAL_STATE,
    TARGET_STATE;
    labels=["Ψ₀", "Ψtgt"],
    xlim=(-500, 500)
)

# ## Analytical Ramp Dynamics

println("Guess fidelity = $(propagate_splitting(; η=(t -> 1.0)))")

# +
"""Special object for efficiently calculating expectation values.

See `map_observables`.
"""
struct PositionMomentumObservables

    theta_op::Diagonal{Float64,Vector{Float64}}
    theta_sq_op::Diagonal{Float64,Vector{Float64}}
    momentum_grid::Vector{Float64}
    coordinate_state::Vector{ComplexF64}
    momentum_state::Vector{ComplexF64}
    vals::Vector{Float64}
    FFT
    IFFT

    function PositionMomentumObservables(; theta_grid::Vector{Float64})
        θ = theta_grid
        dθ = θ[2] - θ[1]
        nθ = length(theta_grid)
        momentum_grid::Vector{Float64} = 2π * fftfreq(nθ, 1 / dθ)
        theta_grid_sq = theta_grid .^ 2
        coordinate_state = zeros(ComplexF64, nθ)
        momentum_state = zeros(ComplexF64, nθ)
        vals = zeros(4)
        FFT = plan_fft(coordinate_state; flags=FFTW.MEASURE)
        IFFT = plan_ifft(momentum_state; flags=FFTW.MEASURE)
        new(
            Diagonal(theta_grid),
            Diagonal(theta_grid_sq),
            momentum_grid,
            coordinate_state,
            momentum_state,
            vals,
            FFT,
            IFFT
        )
    end

end
# -

import QuantumPropagators.Storage: map_observables

# +
"""Calculate values `[⟨θ⟩, σ_θ, ⟨p⟩, σ_p]`.

The values `σ_θ` and `σ_p` are the standard deviations from the expectation
values ⟨θ⟩ and ⟨p⟩
"""
function map_observables(observables::PositionMomentumObservables, Ψ)
    # θ expectation value
    exp_val_theta = real(dot(Ψ, observables.theta_op, Ψ))
    exp_val_theta_sq = real(dot(Ψ, observables.theta_sq_op, Ψ))
    variance_theta::Float64 = exp_val_theta_sq - exp_val_theta^2

    # momentum expectation value
    ϕ = observables.coordinate_state
    ϕ̃ = observables.momentum_state
    p = observables.momentum_grid
    mul!(ϕ̃, observables.FFT, Ψ)
    @. ϕ̃ = p * ϕ̃
    mul!(ϕ, observables.IFFT, ϕ̃)
    exp_val_momentum = real(Ψ ⋅ ϕ)
    @. ϕ̃ = p * ϕ̃
    mul!(ϕ, observables.IFFT, ϕ̃)
    exp_val_momentum_sq = real(Ψ ⋅ ϕ)
    variance_momentum::Float64 = exp_val_momentum_sq - exp_val_momentum^2

    observables.vals[1] = exp_val_theta
    observables.vals[2] = sqrt(variance_theta)
    observables.vals[3] = exp_val_momentum
    observables.vals[4] = sqrt(variance_momentum)
    return observables.vals

end
# -

POSITION_MOMENTUM_OBSERVABLES = PositionMomentumObservables(; theta_grid=THETA_GRID);

map_observables(POSITION_MOMENTUM_OBSERVABLES, INITIAL_STATE)

map_observables(POSITION_MOMENTUM_OBSERVABLES, TARGET_STATE)

expval_dynamics = propagate_splitting(;
    η=(t -> 1.0),
    observables=POSITION_MOMENTUM_OBSERVABLES,
    storage=true,
    ret=:propagation
)

function plot_expval_dynamics(
    tlist, expvals;
    θ₀=0.125π,
    momentum_target=MOMENTUM_TARGET,
    show_standard_deviations=false,
    figsize=(900, 350),
    title="θ and p expectation values",
    margin=15,
    relative_to_theta=zeros(length(tlist)),
    relative_to_momentum=zeros(length(tlist)),
    show_lab_frame_displacement=true,
)
    θ = @view expvals[1, :]
    σ_θ = @view expvals[2, :]
    p = @view expvals[3, :]
    σ_p = @view expvals[4, :]
    θ′ = θ .- θ₀ .- relative_to_theta
    p′ = p .- relative_to_momentum
    if show_standard_deviations
        ax_pos =
            plot(tlist ./ sec, θ′ ./ π; ribbon=σ_θ ./ π, label="", xlabel="time", ylabel="Δθ (π)")
        ax_mom = plot(tlist ./ sec, p′; ribbon=σ_p, label="p(t)", xlabel="time", ylabel="momentum")
    else
        ax_pos = plot(tlist ./ sec, θ′ ./ π; label="", xlabel="time", ylabel="Δθ (π)")
        ax_mom = plot(tlist ./ sec, p′; label="p(t)", xlabel="time", ylabel="momentum")
    end
    if show_lab_frame_displacement
        separation_time = tlist[end]
        displacement = lab_frame_displacement(tlist; separation_time)[end]
        hline!(ax_pos, [displacement / π ], ls=:dash, label="")
    end
    hline!(ax_mom, [momentum_target,], color="black", ls=:dash, label="target")
    plot(ax_pos, ax_mom; size=figsize, plot_title=title, margin=(margin * Plots.px))
end

# +
function lab_frame_displacement(tlist::Vector{Float64}; separation_time=SEPARATION_TIME, ω₀=OMEGA_TARGET)

    dt = tlist[2] - tlist[1]
    ω = discretize(t -> omega_ramp_up(t; w0=ω₀, t_r=separation_time), tlist)
    @assert ω[1] ≈ 0.0
    θ = cumsum(ω) .* dt
    return θ

end

# +
"""The momentum of the moving frame."""
function moving_frame_momentum(ω_vals)
    return -EFFECTIVE_MASS .* ω_vals
end

function moving_frame_momentum(ω, tlist)
    return moving_frame_momentum(discretize(ω, tlist))
end
# -

plot_expval_dynamics(
    TIME_GRID, expval_dynamics;
    show_standard_deviations=true,
    momentum_target=0.0,
    relative_to_momentum=moving_frame_momentum(omega_ramp_up, TIME_GRID)
)

# ## Optimal Control Setup

using QuantumControl

using QuantumControl.Functionals: J_T_sm

includet("./include/guided_amplitude.jl")

includet("./include/pot_scaling_amplitude.jl")

# +
""" Replace amplitudes ω(t) and η(t) in H with guided amplitudes for OCT.
"""
function set_guided_controls(
        H, tlist;
        ϵ_ω=(t->0.0),
        ϵ_η=(t->0.0),
        α_ω=1.0,
        α_η=1.0,
        τ_ω=0.2,
        τ_η=0.2,
    )

    shape(t; α, τ) = α * QuantumControl.Shapes.flattop(
        t,
        T=tlist[end],
        t_rise=τ * tlist[end],
        func=:blackman
    )

    ω_vals::Vector{Float64} = get_controls(H)[1]
    @assert length(ω_vals) == length(tlist) - 1
    ω_ampl = GuidedAmplitude(
        ϵ_ω,
        tlist;
        guide=omega_ramp_up,
        shape=(t -> shape(t; α=α_ω, τ=τ_ω))
    )

    η_vals::Vector{Float64} = get_controls(H)[2]
    @assert length(η_vals) == length(tlist) - 1
    η_ampl = PotScalingAmplitude(
        ϵ_η,
        tlist;
        shape=(t -> shape(t; α=α_η, τ=τ_η))
    )

    return substitute(H, IdDict(ω_vals => ω_ampl, η_vals => η_ampl))

end
# -

@show OMEGA_TARGET;

OCT_HAMILTONIAN = set_guided_controls(
    HAMILTONIAN,
    TIME_GRID;
    α_ω=1.0,# XXX OMEGA_TARGET,  # so that ω_max = 2 ω₀
    α_η=10.0, # XXX 10,  # so that V_max = 2.2 MHz for V₀=0.2 MHz
    ϵ_ω=(t->0.0),
    ϵ_η=(t->1.0),
    τ_ω=0.2,
    τ_η=0.2,
);

# +
objective = Objective(
    initial_state=INITIAL_STATE,
    target_state=TARGET_STATE,
    generator=OCT_HAMILTONIAN,
)

# guess controls (for pulse_options)
ϵ_ω, ϵ_η = get_controls(objective.generator);

problem = ControlProblem(;
    objectives=[objective],
    tlist=TIME_GRID,
    J_T=J_T_sm,
    prop_method=:splitprop,
    verbose=true,
    pulse_options=IdDict(
        ϵ_ω => Dict(:lambda_a => 1e7, :update_shape => t -> 1.0),
        ϵ_η => Dict(:lambda_a => 1e3, :update_shape => t -> 1.0), # XXX :update_shape => t -> 1.0
    ),
    #specrange_method=:manual,
    #check_normalization=true,
    check_convergence=res -> begin
        ((res.J_T < 1e-4) && (res.converged = true) && (res.message = "J_T < 10⁻⁴"))
    end
);
# -

# ## Propagation of Guess Pulses

function get_amplitudes(H::SplitGenerator; as_array=true)
    if as_array
        return (
            (Array(ampl) for ampl in H.T.amplitudes)...,
            (Array(ampl) for ampl in H.V.amplitudes)...
        )
    else
        return (H.T.amplitudes..., H.V.amplitudes)
    end
end

function get_amplitudes(H::SplitGenerator, tlist::Vector{Float64})
    amplitudes = get_amplitudes(H; as_array=true)
    return (discretize(ampl, tlist) for ampl in amplitudes)
end

ω_guess, η_guess = get_amplitudes(OCT_HAMILTONIAN, TIME_GRID);

display(
    plot(
        TIME_GRID ./ μs, ω_guess./ (π/sec);
        xlabel="time (μs)", ylabel="Guess ω(t) (π/sec)", label=""
    )
)

display(
    plot(
        TIME_GRID ./ μs, η_guess .* (POTENTIAL_DEPTH / MHz);
        xlabel="time (μs)", ylabel="Guess V(t) (MHz)", label=""
    )
)

expval_dynamics_guess = propagate_splitting(;
    ω=ω_guess,
    η=η_guess,
    observables=POSITION_MOMENTUM_OBSERVABLES,
    storage=true,
    ret=:propagation
)

plot_expval_dynamics(
    TIME_GRID, expval_dynamics_guess; show_standard_deviations=true,
    momentum_target=0.0,
    relative_to_momentum=moving_frame_momentum(ω_guess)
)

println("Guess fidelity = $(propagate_splitting(; ω=ω_guess, η=(t -> 1.0)))")

# ## Optimization

res = optimize(problem; method=:krotov, iter_stop=500)

H_opt = substitute(
    objective.generator,
    Dict(ϵ => ϵ_opt for (ϵ, ϵ_opt) in zip(res.guess_controls, res.optimized_controls))
);


ω_opt, η_opt = get_amplitudes(H_opt, TIME_GRID);

display(
    plot(
        TIME_GRID ./ μs, ω_opt ./ (π/sec);
        xlabel="time (μs)", ylabel="Opt ω (π/sec)", label=""
    )
)

display(
    plot(
        TIME_GRID ./ μs, (ω_opt .- ω_guess) ./ (π/sec);
        xlabel="time (μs)", ylabel="Δω (π/sec)", label=""
    )
)

display(
    plot(
        TIME_GRID ./ μs, η_opt .* (POTENTIAL_DEPTH / MHz);
        xlabel="time (μs)", ylabel="Opt V(t) (MHz)", label=""
    )
)

display(
    plot(
        TIME_GRID ./ μs, (η_opt .- η_guess) .*  (POTENTIAL_DEPTH / MHz);
        xlabel="time (μs)", ylabel="ΔV(t) (MHz)", label=""
    )
)

# ## Propagation of optimized pulses

expval_dynamics_opt = propagate_splitting(;
    ω=ω_opt,
    η=η_opt,
    observables=POSITION_MOMENTUM_OBSERVABLES,
    storage=true,
    ret=:propagation
)

plot_expval_dynamics(TIME_GRID, expval_dynamics_opt;
    show_standard_deviations=true,
    momentum_target=0.0,
    relative_to_momentum=moving_frame_momentum(ω_opt)
)
