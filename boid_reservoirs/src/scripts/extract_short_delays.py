#!/usr/bin/env python3
"""Extract short-delay R^2 (delays 1–4) for swarm MC at selected N.

Usage (from repo root):
  python boid_reservoirs/src/scripts/extract_short_delays.py \
    --base_yaml boid_reservoirs/config/gpu_scaling_base.yaml \
    --N_values 800 1200 1600 2000 \
    --noise_std 0.05 \
    --seed 0 \
    --steps 3000 \
    --washout 500 \
    --out_md boid_reservoirs/results/r2_short_delays.md \
    --out_json boid_reservoirs/results/r2_short_delays.json

Notes:
- Uses pure MC protocol (poly_degree=1), i.i.d. input generated inside the runner
- Runs one seed per N by default for speed; adjust as needed
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
import numpy as np

# Import runner (same pattern as other scripts)
from boid_reservoirs.src.experiments.run_multi_state import run_multi_state_experiment


def run_one(config_path: Path, N: int, noise_std: float, seed: int, steps: int, washout: int) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    cfg = dict(cfg)
    cfg["N"] = int(N)
    cfg["noise_std"] = float(noise_std)
    cfg["random_seed"] = int(seed)
    cfg["steps"] = int(steps)
    cfg["washout"] = int(washout)
    cfg["poly_degree"] = 1  # pure MC protocol
    # Prefer CPU for portability unless user edits to True
    cfg["use_gpu"] = bool(cfg.get("use_gpu", False))

    # Persist a temporary YAML for the runner
    tmp_yaml = Path(f"/tmp/tmp_cfg_N{N}_noise{noise_std:.2f}_seed{seed}.yaml")
    tmp_yaml.write_text(yaml.safe_dump(cfg))

    res = run_multi_state_experiment(str(tmp_yaml), task="memory_capacity", verbose=False)
    mc_curve = np.array(res["mc_curve"], dtype=float)
    out = {
        "N": int(N),
        "noise_std": float(noise_std),
        "seed": int(seed),
        "r2_d1": float(mc_curve[0]) if mc_curve.size > 0 else float("nan"),
        "r2_d2": float(mc_curve[1]) if mc_curve.size > 1 else float("nan"),
        "r2_d3": float(mc_curve[2]) if mc_curve.size > 2 else float("nan"),
        "r2_d4": float(mc_curve[3]) if mc_curve.size > 3 else float("nan"),
    }
    return out


def write_outputs(rows: list[dict], out_md: Path, out_json: Path) -> None:
    # Markdown table
    lines = ["### R² at short delays (pure MC; one seed per N)\n\n"]
    lines.append("| N | R²₁ | R²₂ | R²₃ | R²₄ |\n")
    lines.append("|---|-----|-----|-----|-----|\n")
    for row in sorted(rows, key=lambda r: r["N"]):
        lines.append(
            f"| {row['N']} | {row['r2_d1']:.3f} | {row['r2_d2']:.3f} | {row['r2_d3']:.3f} | {row['r2_d4']:.3f} |\n"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines))

    # JSON sidecar
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract R² at delays 1–4 for selected N values")
    ap.add_argument("--base_yaml", required=True)
    ap.add_argument("--N_values", nargs="+", type=int, default=[800, 1200, 1600, 2000])
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--washout", type=int, default=500)
    ap.add_argument("--out_md", default="boid_reservoirs/results/r2_short_delays.md")
    ap.add_argument("--out_json", default="boid_reservoirs/results/r2_short_delays.json")
    args = ap.parse_args()

    base_yaml = Path(args.base_yaml)
    rows: list[dict] = []
    for N in args.N_values:
        rows.append(
            run_one(
                base_yaml,
                N=int(N),
                noise_std=float(args.noise_std),
                seed=int(args.seed),
                steps=int(args.steps),
                washout=int(args.washout),
            )
        )

    write_outputs(rows, Path(args.out_md), Path(args.out_json))
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()


