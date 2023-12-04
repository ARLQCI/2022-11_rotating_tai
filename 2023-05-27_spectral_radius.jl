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

# # Comparison of spectral radius for ω₀=10π/s and ω₀=50π/s

using LinearAlgebra
using Revise

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
const TAI_RADIUS = 25.46μm
const N_SITES = 8;
const SEPARATION_TIME = 0.1sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;

includet("./include/rotating_tai.jl")

includet("./include/split_propagator.jl")

includet("./include/free_propagator.jl")

includet("./include/position_momentum_observables.jl")

includet("./include/propagate_scheme.jl")

theta_grid = collect(range(0, 0.25π, length=1024));

H10 = rotating_tai_hamiltonian_coord(;
  theta_grid,
  ω=10π/sec,
  potential_depth=POTENTIAL_DEPTH,
  number_of_sites=N_SITES,
  mass=EFFECTIVE_MASS,
);

eigvals(H10)[1] ./ MHz

eigvals(H10)[end] ./ MHz

H50 = rotating_tai_hamiltonian_coord(;
  theta_grid,
  ω=50π/sec,
  potential_depth=POTENTIAL_DEPTH,
  number_of_sites=N_SITES,
  mass=EFFECTIVE_MASS,
);

eigvals(H50)[1] ./ MHz

eigvals(H50)[end] ./ MHz


