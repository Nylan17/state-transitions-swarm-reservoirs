# State Transitions in Swarm-Based Reservoir Computing

This repository snapshot contains the minimal code, configs, and machine‑readable data needed to reproduce the results and figures reported in the manuscript.

## Citation
Lund T, Adams A, Aubert-Kato N, Ikegami T. State Transitions Unlock Temporal Memory in Swarm-Based Reservoir Computing. PeerJ Computer Science (under review, 2025)

## Environment
- Python ≥3.10
- NumPy, SciPy, scikit‑learn
- Optional: CuPy for GPU (falls back to CPU)

Install:
```bash
pip install -r boid_reservoirs/requirements.txt
```

## Reproduce Main Results
Run all commands from the repository root.

### 1) Ablation at fixed N (pure MC)
Inputs: `docs/claim_map.json` (numerical ground truth) and precomputed per‑condition summaries in `boid_reservoirs/results/`.

Generate the grouped ablation figure (single panel, log y‑axis):
```bash
python -m boid_reservoirs.src.scripts.plot_ablation_grouped \
  --claim_map docs/claim_map.json \
  --out figures/ablation_all_panels.png
```

### 2) Scaling (two‑state, main pipeline)
If using precomputed CSV:
```bash
mkdir -p figures
python -m boid_reservoirs.src.scripts.generate_scaling_figure \
  --csv boid_reservoirs/results/gpu_scaling_complete.csv \
  --out_png figures/mc_scaling.png \
  --out_md boid_reservoirs/results/mc_scaling_summary.md
```

Optionally recompute slope CI from the same CSV:
```bash
python -m boid_reservoirs.src.scripts.compute_slope_ci \
  --csv boid_reservoirs/results/gpu_scaling_complete.csv \
  --out boid_reservoirs/results/mc_slope_ci.md
```

### 3) Pure‑MC slope confirmation (linear readout; no polynomial terms)
```bash
python -m boid_reservoirs.src.scripts.generate_pure_mc_scaling \
  --csv boid_reservoirs/results/gpu_scaling_pure_mc_partial.csv \
  --out_png figures/mc_scaling_pure.png \
  --out_md boid_reservoirs/results/mc_scaling_pure_summary.md
```

### 4) Short‑delay R² tables (compact check)
```bash
python -m boid_reservoirs.src.scripts.extract_short_delays \
  --base_yaml boid_reservoirs/config/multi_state_defaults.yaml \
  --N_values 800 1200 1600 2000 \
  --noise_std 0.05 \
  --seed 0 \
  --steps 3000 \
  --washout 500 \
  --out_md boid_reservoirs/results/r2_short_delays.md \
  --out_json boid_reservoirs/results/r2_short_delays.json
```

### 5) ESN baseline (optional context)
```bash
python -m boid_reservoirs.src.scripts.plot_esn_comparison \
  --json boid_reservoirs/results/comparison/esn_results.json \
  --out_png figures/esn_mc_summary.png \
  --out_md boid_reservoirs/results/comparison/esn_summary.md
```

### 6) Ablation per-seed CSV exporter
```bash
python -m boid_reservoirs.src.scripts.export_ablation_per_seed \
  --claim_map docs/claim_map.json \
  --out boid_reservoirs/results/ablation_per_seed.csv
```

## Regenerating Raw Runs (optional)
You can regenerate per‑seed memory capacity runs (pure MC) and aggregate, but this is not required if you trust the provided CSVs.

Example (pure‑MC per‑seed):
```bash
python -m boid_reservoirs.src.experiments.run_mc boid_reservoirs/config/single_state_mc.yaml  # single‑state
python -m boid_reservoirs.src.experiments.run_multi_state \
  --config boid_reservoirs/config/multi_state_defaults.yaml \
  --task memory_capacity --quiet  # two‑state
```
Aggregate into CSVs using your preferred script or a small adapter; ensure the final CSV schema matches `gpu_scaling_complete.csv` / `gpu_scaling_pure_mc_partial.csv`.

## File Map
- Code (core):
  - `boid_reservoirs/src/boids/core.py`
  - `boid_reservoirs/src/boids/multi_state_core.py`
  - `boid_reservoirs/src/boids/features.py`
  - `boid_reservoirs/src/boids/multi_state_features.py`
  - `boid_reservoirs/src/boids/input_couplers.py`
  - `boid_reservoirs/src/rc/memory_capacity.py`
  - `boid_reservoirs/src/rc/ridge.py`
  - `boid_reservoirs/src/rc/metrics.py`
- Scripts (repro):
  - `boid_reservoirs/src/scripts/generate_scaling_figure.py`
  - `boid_reservoirs/src/scripts/compute_slope_ci.py`
  - `boid_reservoirs/src/scripts/generate_pure_mc_scaling.py`
  - `boid_reservoirs/src/scripts/extract_short_delays.py`
  - `boid_reservoirs/src/scripts/plot_ablation_grouped.py`
  - `boid_reservoirs/src/scripts/summarize_esn_mc_scaling.py` (optional)
  - `boid_reservoirs/src/scripts/plot_esn_comparison.py` (optional)
- Experiments:
  - `boid_reservoirs/src/experiments/run_mc.py`
  - `boid_reservoirs/src/experiments/run_multi_state.py` (optional tasks)
- Configs:
  - `boid_reservoirs/config/multi_state_defaults.yaml`
  - `boid_reservoirs/config/single_state_mc.yaml`
- Data (machine‑readable):
  - `boid_reservoirs/results/gpu_scaling_complete.csv`
  - `boid_reservoirs/results/gpu_scaling_pure_mc_partial.csv`
  - `boid_reservoirs/results/ablation_s1_vs_s2_N800.md`
  - `boid_reservoirs/results/ablation_s1_vs_s2_N1200.md`
  - `boid_reservoirs/results/ablation_s1_vs_s2_N1600.md`
  - `boid_reservoirs/results/comparison/esn_summary.md` (optional)
  - `boid_reservoirs/results/mg17_verification_singlestate.json`
  - `boid_reservoirs/results/mg17_verification_multistate.json`
  - `docs/claim_map.json`
  - `boid_reservoirs/data/mackey_glass_17.txt` (for MG verification)
- Seeds:
  - `docs/seeds.yaml`

## License
The source code in this repository is licensed under the **MIT License**. See the `LICENSE` file for more details.
The data and figures are licensed under the **Creative Commons Attribution 4.0 International License** ([CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)).