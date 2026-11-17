# DPF-PSO: dynamic-programming path fusion for swarm-based 3-D UAV path planning

Research code, raw results and manuscript sources (third revision) prepared
for ICAISI 2027 (full-paper deadline 1 Feb 2027, https://icaisi.net/).
Read `notes/design_log.md` first: it records how the method and the
experimental protocol were fixed and which data each decision was based on.

## Layout

| Path | Content |
|---|---|
| `src/env.py` | Terrain (bilinear, exact segment clearance), threats, scenarios S1–S6, held-out generator H01–H40, public scenario C1, slot and spherical encodings, decomposable cost model |
| `src/dpf.py` | DP path fusion (exact, with exact pruning), global/local fusion, FE-based host hook |
| `src/algorithms.py` | PSO, SPSO, GWO, WOA, HHO, DE, L-SHADE, CMA-ES, IDP, GLS, DPF-PSO, plug-in and ablation variants, tuned configurations (`-T`) |
| `src/core.py` | FE-budget evaluator |
| `src/run_experiments.py` | Parallel runner (`--tag --runs --instances --algos --budget-mult --mult --params --skip-existing`) |
| `src/tune.py` | Equal-budget random-search tuning on S1–S4 |
| `src/timing_benchmark.py`, `src/cpuinfo_win.py` | CPU-time benchmark pinned to the performance cores; equal-time multipliers |
| `src/stats2.py`, `src/analysis2.py` | Statistics (medians, gaps, A12, Holm, rank-sum and signed-rank tests), tables, figures |
| `src/clearance_ref.py`, `src/check_clearance.py` | Independent exact clearance implementation and its verification |
| `src/check_ceiling.py` | Exact maximum height above ground (clearance on the mirrored terrain) against the independent implementation |
| `src/sanity_checks.py` | DP = brute force = full evaluation; budgets respected (`--band`: the same for the altitude-band variant) |
| `src/check_pruning.py` | Pruned DP bit-identical to dense DP; timing (run on an idle machine) |
| `src/verify_paths.py` | Independent safety check of every final path (checks the altitude band along the segments for experiments run with `band_along`) |
| `data/get_dem.py` | Downloads the public DEM of scenario C1 (not redistributed) and checks its SHA-256 |
| `paper/build2.py` | Fills `paper_template2.md` and `supplement_template.md` with numbers computed from the results; builds DOCX and PDF |
| `run_all.ps1` | Full experiment pipeline of the second revision |
| `run_rev3.ps1` | Third-revision experiments: CMA-ES tuning, altitude band along the segments (`heldout_band`), held-out ablation, held-out equal CPU time (`heldout_eqtime`) |
| `results/<tag>/` | Raw results, one `.npz` per instance and algorithm |
| `notes/` | Design log, verification outputs, reference checks, scratch checks |

## Reproduce

```powershell
python data/get_dem.py                     # public DEM for C1
python src/sanity_checks.py                # DP exactness, budgets
python src/sanity_checks.py --band         # the same for the altitude-band variant
python src/check_clearance.py              # exact clearance vs independent implementation
python src/check_ceiling.py                # exact maximum height vs independent implementation
python src/tune.py --workers 20            # equal-budget tuning on S1-S4 (~45 min)
powershell -File run_all.ps1               # timing + all experiments (~8 h on 24 threads)
powershell -File run_rev3.ps1              # third-revision experiments (~5 h on 12 workers)
python src/analysis2.py                    # summary.json, tables, figures
python src/verify_paths.py main heldout public eqtime n30 budget5 b2 heldout_band heldout_eqtime
python src/check_pruning.py                # idle machine
python paper/build2.py                     # paper and supplement (needs pandoc and MS Word)
```

`paper/build2.py` refuses to build if a data-dependent sentence of the
templates no longer matches the results (`<!--ASSERT ...-->` guards); use
`--draft` to see the text with missing values marked.  Scripts in `notes/`
are checks used during the study (CMA-ES restart thresholds, reference
metadata, quick looks at results); they are not needed to reproduce the
paper.

Requirements: Python 3.13, NumPy 2.2, SciPy, Matplotlib, pycma 4.5, psutil,
Pillow, python-docx, pywin32 and Microsoft Word (PDF export), pandoc 2.12,
PyMuPDF (page previews).  Seeds: tuning 30270000 + r, final runs
20271101 + r (run r of every method uses the same seed), timing
40270000 + r, held-out scenario k from 90210 + k.
