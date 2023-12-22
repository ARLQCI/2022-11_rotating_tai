.PHONY: ipynb

NOTEBOOKFILES = \
    2022-11-02_test_split_propagator.ipynb \
    2022-11-04_full_scheme.ipynb \
    2022-11-07_test_integrated_amplitude.ipynb \
    2022-11-08_fast_tai.ipynb \
    2022-11-10_slow_tai_rotating_frame.ipynb \
    2022-11-12_fast_tai_rotating_frame.ipynb \
    2022-11-14_fast_tai_frame_comparison.ipynb \
    2022-11-17_slow_tai_rotating_frame_100ms_loop.ipynb \
    2022-11-17_superfast_tai_rotating_frame.ipynb \
    2022-11-18_superfast_tai_OCT.ipynb \
    2022-12-23_compressed_tai_OCT.ipynb \
    2023-01-05_map_splitting.ipynb \
    2023-01-25_OCT_tr=0.1μs_V0=0.1MHz.ipynb \
    2023-01-30_OCT_tune_potential.ipynb \
    2023-02-01_map_analysis.ipynb \
    2023-02-06_OCT_tr=100μs_V0=2.1MHz.ipynb \
    2023-02-06_OCT_tr=150μs_V0=0.2MHz.ipynb \
    2023-02-06_OCT_tr=150μs_V0=2.1MHz.ipynb \
    2023-02-07_continued_evolution.ipynb \
    2023-02-20_map_analysis_omega_100πps.ipynb \
    2023-02-20_omega_100πps_gridpoints.ipynb \
    2023-02-28_wigner_analysis.ipynb \
    2023-03-09_OCT_tr=150μs_V0=0.2MHz_varpot.ipynb \
    2023-03-24_guess_full_scheme.ipynb \
    2023-03-27_test_freeprop.ipynb \
    2023-04-05_adiabatic_lab_frame.ipynb \
    2023-04-17_adiabatic_lab_frame_initialize_with_Ω.ipynb \
    2023-04-20_optimized_full_scheme.ipynb \
    2023-04-24_harmonic_dynamics.ipynb \
    2023-05-16_guess_full_scheme_R26μ.ipynb \
    2023-05-16_guess_full_scheme_R=26μm_ω=50πps.ipynb \
    2023-05-16_map_splitting_R26μ_50pps.ipynb \
    2023-05-16_map_splitting_R26μ.ipynb \
    2023-05-17_adiabatic_full_scheme_R=26μm_ω=50πps.ipynb \
    2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps.ipynb \
    2023-05-17_optimized_full_scheme_R=26μm_ω=50πps.ipynb \
    2023-05-18_harmonic_dynamics_R=26μm_ω=50πps.ipynb \
    2023-05-25_adiabatic_full_scheme_R=26μm_ω=10πps.ipynb \
    2023-05-27_spectral_radius.ipynb \
	2023-12-04_optimized_full_scheme_robustness.ipynb

ipynb: $(NOTEBOOKFILES)

%.ipynb : | %.jl
	JULIA_NUM_THREADS=1 jupytext --to notebook --execute $(*).jl
