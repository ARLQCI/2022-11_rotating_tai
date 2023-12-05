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

using FromFile
using Test
using LinearAlgebra
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using QuantumPropagators: propagate

@from "include/free_propagator.jl" import FreePropWrk, freeprop!

const N = 100;

Ψ₀ = random_state_vector(N);

H = random_matrix(N; hermitian=false);

wrk = FreePropWrk(Ψ₀, H);

dt = 3.5

Ψ₁ = copy(Ψ₀)
freeprop!(Ψ₁, dt, wrk);

Ψ₂ = exp(-1im * H * dt) * Ψ₀;

λ, U = eigen(H);

norm(Ψ₀ - (U * (inv(U) * Ψ₀)))

norm(Ψ₀ - (U * (U' * Ψ₀)))

@test norm(Ψ₁ - Ψ₂) < 1e-12

Ψ₃ = propagate(Ψ₀, H, [0.0, dt]; method=:freeprop);

@test norm(Ψ₁ - Ψ₃) < 1e-12

Ψ₄ = propagate(Ψ₀, H, [0.0, dt/2, dt]; method=:freeprop);

@test norm(Ψ₁ - Ψ₄) < 1e-12
