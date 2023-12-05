---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Markdown
    language: markdown
    name: markdown
---

# Index


**2023-01-05** — Map of splitting fidelity for $\omega_0 = 10\pi$/s

* [2023-01-05_map_splitting.ipynb](2023-01-05_map_splitting.ipynb)

This includes full scheme dynamics / constrast analysis


**2023-01-30** — Proof of concept of tuning both amplitude and potential

* [2023-01-30_OCT_tune_potential.ipynb](2023-01-30_OCT_tune_potential.ipynb)


**2023-02-01** — Analysis of dynamics for points of splitting fidelity map ($\omega_0 = 10\pi$/s)

* [2023-02-01_map_analysis.ipynb](2023-02-01_map_analysis.ipynb)


**2023-02-06** — Set of optimal control attempts - Fixed V₀

* [2023-02-06_OCT_tr=100μs_V0=2.1MHz.ipynb](2023-02-06_OCT_tr=100μs_V0=2.1MHz.ipynb) — point 6 on map analysis
* [2023-02-06_OCT_tr=150μs_V0=0.2MHz.ipynb](2023-02-06_OCT_tr=150μs_V0=0.2MHz.ipynb) – point 5 on map analysis
* [2023-02-06_OCT_tr=150μs_V0=2.1MHz.ipynb](2023-02-06_OCT_tr=150μs_V0=2.1MHz.ipynb) – point 5 on map analysis


**2023-02-07** — Can we improve the map by "waiting for separation"?

* [2023-02-07_continued_evolution.ipynb](2023-02-07_continued_evolution.ipynb) — look at oscillations if we wait a little bit after the splitting:
    * Separation is always smaller than the width of the wave packet
    * *Fidelity* is constant
    * We're just rotating in phase space → Waiting will not change the map!


**2023-02-20** – Analysis for $\omega_0 = 100\pi$/s

Because for $\omega_0 = 10\pi$/s, the initial state and the target state have significant overlap, it would be interesting to look at a value of $\omega_0$ where the two states are fully separated in momentum space. This can be achieved by increasing $\omega_0$ by one order of magnitude to $\omega_0 = 100\pi$/s.

* [2023-02-20_omega_100πps_gridpoints.ipynb](2023-02-20_omega_100πps_gridpoints.ipynb) — determine how many grid points in $\theta$ we need to represent a momentum of 100π/s. Result: 1024 points is okay for map, 2028 points for optimal control if $\omega(t)$ stays below 1000π/sec.
* [2023-02-20_map_analysis_omega_100πps.ipynb](2023-02-20_map_analysis_omega_100πps.ipynb) — create a map of splitting fidelity

    * Map that does not include the "overlap effect"!


**2023-02-28** — Exploring the Wigner Representation

* [2023-02-28_wigner_analysis.ipynb](2023-02-28_wigner_analysis.ipynb) — Implement Wigner plot (testing)

Open questions:

* What are the proper units for "momentum"?
* How do we justify the `ratio` parameter?


**2023-03-09** — Setting up a possible optimizations for the paper (varying both ω(t) and V(t))

* [2023-03-09_OCT_tr=150μs_V0=0.2MHz_varpot.ipynb](2023-03-09_OCT_tr=150μs_V0=0.2MHz_varpot.ipynb) — Optimize with potential ramping up from 0.2MHz to 2.2 MHz

Contains code for plotting expectation values entirely in the dynamic frame (momentum)

We find that scaling the potential introduces breathing in the wave package. The optimization easily gets to the correct expectation values, but can't correct the "squeezing". Possible future remedy: choose potential / timing so that we have exactly one period in the breathing.


**2023-03-23 - 2023-04-26** — Full scheme guess and optimized dynamics (R = 42μm)

For draft 1 of the paper, we decide to look at either adiabatic dynamics (`t_r=100ms, V0=2.2MHz`), or non-adiabatic dynamics (`t_r=150μs`, `V0=0.2MHz`), un-optimized and optimized. The optimization is in [2023-02-06_OCT_tr=150μs_V0=0.2MHz.ipynb](2023-02-06_OCT_tr=150μs_V0=0.2MHz.ipynb). These notebooks use a reimplementation of the full scheme in `./include/propagate_scheme.jl` and a new custom "free propagator" (`./include/free_propagator.jl`)

* [2023-03-24_guess_full_scheme.ipynb](2023-03-24_guess_full_scheme.ipynb) — full dynamics of the guess pulse for `t_r=150μs`, `V0=0.2MHz`
* [2023-03-27_test_freeprop.ipynb](2023-03-27_test_freeprop.ipynb) — testing the new free propagator
* [2023-04-05_adiabatic_lab_frame.ipynb](2023-04-05_adiabatic_lab_frame.ipynb) — full dynamics for the adiabatic parameters
* [2023-04-17_adiabatic_lab_frame_initialize_with_Ω.ipynb](2023-04-17_adiabatic_lab_frame_initialize_with_Ω.ipynb)  — full dynamics for the adiabatic parameters — full dynamics for the adiabatic parameters, but for the Sagnac response, the eigenvalues are calculated with whatever the value of $\Omega$ is.
* [2023-04-20_optimized_full_scheme.ipynb](2023-04-20_optimized_full_scheme.ipynb) — Full dynamics of the optimized scheme at `t_r=150μs`, `V0=0.2MHz`
* [2023-04-24_harmonic_dynamics.ipynb](2023-04-24_harmonic_dynamics.ipynb) — Dynamics for a harmonic oscillator trap, for both the adiabatic and nonadiabatic parameters
* [2023-04-26_sympy_recombination.ipynb](2023-04-26_sympy_recombination.ipynb) — analytic derivation of what happens during recombination


**2023-05-16 - 2023-04-17** — Revisions for paper submission

Most importantly, we adjusted the radius to 26μm.

* [2023-05-16_map_splitting_R26μ.ipynb](2023-05-16_map_splitting_R26μ.ipynb) — Landscape map at the new values of R, with ω₀=10π/sec
* [2023-05-16_map_splitting_R26μ_50pps.ipynb](2023-05-16_map_splitting_R26μ_50pps.ipynb) — Same as above, but with faster ω₀
* [2023-05-16_guess_full_scheme_R=26μm_ω=50πps.ipynb](2023-05-16_guess_full_scheme_R=26μm_ω=50πps.ipynb)
* [2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps.ipynb](2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps.ipynb)
* [2023-05-17_adiabatic_full_scheme_R=26μm_ω=50πps.ipynb](2023-05-17_adiabatic_full_scheme_R=26μm_ω=50πps.ipynb)
* [2023-05-17_optimized_full_scheme_R=26μm_ω=50πps.ipynb](2023-05-17_optimized_full_scheme_R=26μm_ω=50πps.ipynb)
* [2023-05-18_harmonic_dynamics_R=26μm_ω=50πps.ipynb](2023-05-18_harmonic_dynamics_R=26μm_ω=50πps.ipynb) — comparing the dynamics of the cos potential with a harmonic potential
* [2023-05-25_adiabatic_full_scheme_R=26μm_ω=10πps.ipynb](2023-05-25_adiabatic_full_scheme_R=26μm_ω=10πps.ipynb)
* [2023-05-27_spectral_radius.ipynb](2023-05-27_spectral_radius.ipynb)

**2023-12-04** — Further exploration based on referee reports

* [2023-12-04_optimized_full_scheme_robustness.ipynb](2023-12-04_optimized_full_scheme_robustness.ipynb) — How does the contrast change if we take the optimal solution obtained at V₀=0.2MHz and run it at a 10% variation of V₀. Answer: not much changes, not even if we use the original eigenstate as the original state (worst case for V₀ that varies over time)
