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
#     display_name: Julia 1.8 (auto threads)
#     language: julia
#     name: julia-1.8-multithread
# ---

# #  OCT tuning both ω(t) and V(θ)

# This explores OCT for fastest separation time `t_r=0.1μs`, and the weakes potential, `V0=0.1MHz` if we allow both the the ramp $\omega(t)$ and the potential $V(\theta)$ to vary

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
const SEPARATION_TIME = 0.1μs;
const LOOP_TIME = 900ms;
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 0.1MHz;

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

# ## Hamiltonian

using QuantumPropagators.Controls: discretize, discretize_on_midpoints

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

function propagate_splitting(
    separation_time=SEPARATION_TIME,
    potential_depth=POTENTIAL_DEPTH;
    omega_target=OMEGA_TARGET,
    number_of_sites=N_SITES,
    mass=EFFECTIVE_MASS,
    ret=:fidelity,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    theta_max=0.25π,
    theta_steps=1024,
    scale_potential=nothing,
    kwargs...
)
    nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, separation_time, length=nt))
    ω_func(t) = omega_ramp_up(t; w0=omega_target, t_r=separation_time)
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    if !isnothing(scale_potential)
        scale_potential = discretize_on_midpoints(scale_potential, tlist)
    end
    Ĥ = rotating_tai_hamiltonian(;
        tlist,
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        scale_potential,
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

HAMILTONIAN, TIME_GRID = propagate_splitting(scale_potential=(t -> 1.0); ret=:system);

HAMILTONIAN.V

HAMILTONIAN.V

length(get_controls(HAMILTONIAN))

Ĥ₀ = evaluate(HAMILTONIAN, TIME_GRID, 1)

evaluate!(Ĥ₀, HAMILTONIAN, TIME_GRID, 1)

# ## Initial state

INITIAL_STATE, THETA_GRID = propagate_splitting(; ret=:initial_state);

# ## Target state

TARGET_STATE, _ = propagate_splitting(; ret=:target);

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

# ## Guess Dynamics

println("Guess fidelity = $(propagate_splitting())")

includet("./include/position_momentum_observables.jl")

POSITION_MOMENTUM_OBSERVABLES = PositionMomentumObservables(; theta_grid=THETA_GRID);

map_observables(POSITION_MOMENTUM_OBSERVABLES, INITIAL_STATE)

map_observables(POSITION_MOMENTUM_OBSERVABLES, TARGET_STATE)

expval_dynamics = propagate_splitting(;
    observables=POSITION_MOMENTUM_OBSERVABLES,
    storage=true,
    ret=:propagation
)

# +
function get_expval_dynamics(;
    separation_time=SEPARATION_TIME,
    theta_max=0.25π,
    theta_steps=1024,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    show=true,
    show_standard_deviations=false,
    kwargs...
)

    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    observables = PositionMomentumObservables(; theta_grid=θ)
    expvals = propagate_splitting(;
        separation_time,
        theta_max,
        theta_steps,
        timesteps_per_microsec,
        minimum_timesteps,
        observables,
        storage=true,
        ret=:propagation,
        kwargs...
    )

    if show
        nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
        tlist = collect(range(0, separation_time, length=nt))
        plot_expval_dynamics(tlist, expvals; show_standard_deviations)
    else
        return expvals
    end

end
# -

function plot_expval_dynamics(tlist, expvals; show_standard_deviations=false)
    θ = @view expvals[1, :]
    σ_θ = @view expvals[2, :]
    p = @view expvals[3, :]
    σ_p = @view expvals[4, :]
    if show_standard_deviations
        ax_pos =
            plot(tlist, θ ./ π; ribbon=σ_θ ./ π, label="", xlabel="time", ylabel="θ (π)")
        ax_mom = plot(tlist, p; ribbon=σ_p, label="", xlabel="time", ylabel="momentum")
    else
        ax_pos = plot(tlist, θ ./ π; label="", xlabel="time", ylabel="θ (π)")
        ax_mom = plot(tlist, p; label="", xlabel="time", ylabel="momentum")
    end
    plot(ax_pos, ax_mom)
end

get_expval_dynamics()

# ## Optimal Control

using QuantumControl

using QuantumControl.Functionals: J_T_sm

includet("./include/guided_amplitude.jl")

update_shape(t) = QuantumControl.Shapes.flattop(
    t,
    T=TIME_GRID[end],
    t_rise=0.2 * TIME_GRID[end],
    func=:blackman
)

function set_guided_control(H, tlist)
    ω_vals::Vector{Float64} = get_controls(H)[1]
    @assert length(ω_vals) == length(tlist) - 1
    control = GuidedAmplitude(t -> 0.0, tlist; guide=omega_ramp_up, shape=update_shape)
    return substitute(H, IdDict(ω_vals => control))
end

objective = Objective(
    initial_state=INITIAL_STATE,
    target_state=TARGET_STATE,
    generator=set_guided_control(HAMILTONIAN, TIME_GRID)
)

plot(get_controls(objective.generator)[1])

plot(get_controls(objective.generator)[2])

# +
δω, η = get_controls(objective.generator);

problem = ControlProblem(;
    objectives=[objective],
    tlist=TIME_GRID,
    J_T=J_T_sm,
    prop_method=:splitprop,
    verbose=true,
    pulse_options=IdDict(
        δω => Dict(:lambda_a => 1000, :update_shape => t -> 1.0),
        η => Dict(:lambda_a => 1e-4, :update_shape => update_shape),
    ),
    #specrange_method=:manual,
    #check_normalization=true,
    check_convergence=res -> begin
        ((res.J_T < 1e-4) && (res.converged = true) && (res.message = "J_T < 10⁻⁴"))
    end
);
# -

res = @optimize_or_load(
    "./data/2023-01-01_OCT_tune_potential_opt_krotov.jld2",
    problem; method=:krotov, iter_stop=1_000_000, force=false,
)

# I tried for more iteration, but nothing really happens

plot(res.optimized_controls[1])

plot(res.optimized_controls[2])

OMEGA_TARGET

ω_opt = discretize(
    Array(GuidedAmplitude(res.optimized_controls[1], TIME_GRID; guide=omega_ramp_up, shape=update_shape)),
    TIME_GRID
)

plot(TIME_GRID ./ μs, ω_opt / (2π / sec))

res_grape = optimize(problem; method=:grape, prop_method=:cheby, iter_stop=1)

# ## Optimal Control with handholding

includet("./include/pot_scaling_amplitude.jl")

# +
function set_guided_controls(H, tlist; omega_scale=1.0, pot_scale=1.0, pot_scale_guess=(t->0.0))
    
    shape(t; α) = α * QuantumControl.Shapes.flattop(
        t,
        T=tlist[end],
        t_rise=0.2 * tlist[end],
        func=:blackman
    )
    
    ω_vals::Vector{Float64} = get_controls(H)[1]
    @assert length(ω_vals) == length(tlist) - 1
    ω_ampl = GuidedAmplitude(
        t -> 0.0,
        tlist;
        guide=omega_ramp_up,
        shape=(t -> shape(t, α=omega_scale))
    )
    
    η_vals::Vector{Float64} = get_controls(H)[2]
    @assert length(η_vals) == length(tlist) - 1
    η_ampl = PotScalingAmplitude(
        pot_scale_guess,
        tlist;
        shape=(t -> shape(t; α=pot_scale))
    )
    
    return substitute(H, IdDict(ω_vals => ω_ampl, η_vals => η_ampl))

end
# -

pot_scale_guess(t) = sqrt(max(0.0, QuantumControl.Shapes.blackman(t, 0, TIME_GRID[end])))

plot(TIME_GRID, pot_scale_guess)

# +
objective = Objective(
    initial_state=INITIAL_STATE,
    target_state=TARGET_STATE,
    generator=set_guided_controls(
        HAMILTONIAN,
        TIME_GRID;
        omega_scale=OMEGA_TARGET,
        pot_scale=2.2MHz,
        pot_scale_guess
    )
)

δω, η = get_controls(objective.generator);

problem = ControlProblem(;
    objectives=[objective],
    tlist=TIME_GRID,
    J_T=J_T_sm,
    prop_method=:splitprop,
    verbose=true,
    pulse_options=IdDict(
        δω => Dict(:lambda_a => 1, :update_shape => t -> 0.0),
        η => Dict(:lambda_a => 1e5, :update_shape => t -> 1.0),
    ),
    #specrange_method=:manual,
    #check_normalization=true,
    check_convergence=res -> begin
        ((res.J_T < 1e-4) && (res.converged = true) && (res.message = "J_T < 10⁻⁴"))
    end
);
# -

res2 = @optimize_or_load(
    "./data/2023-01-01_OCT_tune_potential_opt_krotov_handholding.jld2",
    problem; method=:krotov, iter_stop=100, force=true,
)

plot(res2.optimized_controls[1])

plot(res2.optimized_controls[2])

H_opt = substitute(
    objective.generator,
    Dict(ϵ => ϵ_opt for (ϵ, ϵ_opt) in zip(res2.guess_controls, res2.optimized_controls))
);

function get_amplitudes(H::SplitGenerator)
    amplitudes =  (H.T.amplitudes..., H.V.amplitudes...)
    return Array.(amplitudes)
end

plot(TIME_GRID, discretize(get_amplitudes(H_opt)[2]./MHz, TIME_GRID))
