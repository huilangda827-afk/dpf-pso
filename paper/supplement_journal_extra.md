
# S12. Pre-Registered Test on H41–H80

The plan of this test was written into the design log (`notes/design_log.md`, section 15) before the scenarios H41–H80 were generated and before any run on them. Scenario H$k$ is generated from seed 90210 + $k$ as H01–H40; the seeds 90251–90290 had never been used. Methods: SPSO, DE, IDP, DPF-PSO and the tuned SPSO and DPF-PSO of Section 5.3 (tuned on S1–S4 for the original model); 10 runs per method and scenario with the seeds of all other final runs; budget $3000n$ FEs. Family T (4 tests): tuned SPSO against tuned DPF-PSO under the original and under the band model, Holm over the four tests. Family D (6 tests): DPF-PSO against SPSO, DE and IDP under the band model. Family R (6 tests, added before the results were looked at): the same under the original model. Criterion for "confirmed": tuned DPF-PSO significantly better than tuned SPSO under the band model for both $n$, and the original-model comparison with the same direction as on H01–H40. Outcome: the band finding was @@band_status@@ (Section 6.3).

: TABLE S12a. Family T on H41–H80: Tuned SPSO Against Tuned DPF-PSO Under the Original Model (Reference: Tuned DPF-PSO)

@@S_H2_T@@

: TABLE S12b. Family T on H41–H80: Tuned SPSO Against Tuned DPF-PSO Under the Band Model (Reference: Tuned DPF-PSO)

@@S_H2_BT@@

: TABLE S12c. Family D on H41–H80: Default Configurations Under the Band Model

@@S_H2_D@@

: TABLE S12d. Family R on H41–H80: Default Configurations Under the Original Model

@@S_H2_R@@

::: {custom-style="TableNote"}
Columns as in Table S2. Families T and D were pre-registered; family R was added before any result of these runs was looked at (design log, section 15, addendum).
:::

: TABLE S12e. Height Above Ground Along the Final Paths on H41–H80 (Original Model)

@@S_AGL_H2@@

: TABLE S12f. Feasible Runs Under the Band Model on H41–H80

@@S_H2_BAND_FEAS@@

# S13. Lattice-DP Baseline

GridDP (Section 5.2) places, in every slot, a regular lattice of $n_u$ lateral offsets × 3 heights × 3 along-track offsets and solves one dynamic program over it; $n_u$ is the largest value whose segment evaluations $2K+(n-1)K^2$, $K=9n_u$, fit into the budget of $3000n(n+1)$ segments ($n_u=$ @@grid_nu10@@, $K=$ @@grid_K10@@, @@grid_fe10@@% of the budget for $n=10$; $n_u=$ @@grid_nu20@@, $K=$ @@grid_K20@@, @@grid_fe20@@% for $n=20$). It is deterministic, so one run per instance is reported against the 10 runs of DPF-PSO; all @@grid_feas@@ of its paths were feasible. The baseline was added after the held-out results of the other methods were known (design log, section 16).

: TABLE S13. DPF-PSO Against the Lattice DP on H01–H40 and H41–H80 (One Family of @@hg_ntests@@ Tests)

@@S_GRID@@

# S14. CPU-Time Convergence

Runs of `src/timing_curves.py`: one worker per performance core, each pinned to one logical processor, methods interleaved, process CPU time recorded at the 101 FE checkpoints; @@tc_n_scen20@@ held-out scenarios per $n$, 5 runs per method and scenario (seeds 50270000 + $r$). Table 7 of the main text reports the ratio of the CPU time at which the median best-so-far curve of DPF-PSO first reaches the final median cost of a method to that method's median CPU time; Table S14 lists the median CPU times per run.

: TABLE S14. Median CPU Time per Run (s) in the Timing-Curve Experiment

@@S_TC_CPU@@

# S15. Second Round of Additions (Design Log Section 18)

: TABLE S15a. Equal CPU Time on H01–H40, All Baselines (Detail of Table 3)

@@S_HO_EQ@@

: TABLE S15b. Plug-In on H01–H40: Host with DPF Against the Host Alone

@@S_PLUG_HO@@

: TABLE S15c. Spherical Encoding Against the Slot Encoding (DE and CMA-ES, Original Model)

@@S_SPH_VS_SLOT@@

: TABLE S15d. DE and CMA-ES on the Spherical Encoding Against DPF-PSO, Original Model

@@S_SPH@@

: TABLE S15e. DE and CMA-ES on the Spherical Encoding Against DPF-PSO, Band Model

@@S_SPH_BAND@@

: TABLE S15f. Multi-Start Powell Against DPF-PSO on H01–H40

@@S_MSP@@

: TABLE S15g. Weight Sensitivity $b_4=10$ on S1–S6 (10 Runs; Columns as in Table S5a)

@@S_B4@@
